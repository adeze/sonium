//! Deterministic product interior-acoustic mesh construction.
//!
//! This module is the first product-mesh lowering used by the full 3-D
//! reference path. It supports a sealed rectangular cavity with planar
//! cabinet-wall driver patches, a bounded Cartesian lowering for declared
//! panel-rib braces, and a body-fitted PLC for axis-aligned and bounded
//! oriented single-driver apertures, a straight
//! front port, and single boundary-touching braces. The Cartesian and
//! body-fitted lowerings fail closed when their declared solid geometry cannot
//! be represented at the requested resolution.
//!
//! The returned marker and patch contract is intentionally independent from
//! the legacy cylindrical-fixture markers.  Future body-fitted port, brace,
//! waveguide, and exterior-BEM lowerings extend the same typed registry rather
//! than reintroducing a single anonymous aperture marker.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use sonium_fem::mesh::{
    BoundaryType, ElementType, Mesh as FemMesh, Point, SurfaceFacet, TetrahedralizeConfig,
    box_mesh_tetrahedra, tetrahedralize_surface_with_facets,
};
use sonium_mesher::mesh::triangulate::triangulate_convex_polygon_with_holes;

use crate::{
    cabinet_design::{BraceDesign, CabinetDesign, PanelAxis, PanelId, PortDesign},
    product_design::{DriverMountingFace, ProductDesign, ProductDesignError},
};

const MARKER_VERSION: u64 = 1;
const WALL_MARKER_BASE: i32 = 20_000;
const DRIVER_REAR_MARKER_BASE: i32 = 21_000;
const PORT_MOUTH_MARKER: i32 = 22_000;
const PORT_WALL_MARKER: i32 = 22_001;
const BRACE_WALL_MARKER_BASE: i32 = 23_000;
const WAVEGUIDE_WALL_MARKER_BASE: i32 = 24_000;
const FRONT_PLANE_TOLERANCE_M: f64 = 1.0e-10;
// A 24-sided inscribed driver disk remains within the public 2% circular-area
// acceptance tolerance even when the nominal edge policy is coarser.
const MIN_BODY_FITTED_RING_SEGMENTS: usize = 24;

/// Fully resolved geometric contract for a straight front-panel bass-reflex
/// duct.
///
/// The product coordinate convention is deliberately encoded here rather than
/// rediscovered independently by the mesher and coupling layers: the baffle
/// is `z = 0`, the enclosure lies at negative `z`, and a positive port-mouth
/// velocity is directed towards the exterior.  `internal_exit_center_m` is
/// the end of the bore that opens into the enclosure, not an acoustic end
/// correction location.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ResolvedPortGeometry {
    /// Centre of the exterior mouth in product coordinates.
    pub mouth_center_m: [f64; 3],
    /// Centre of the bore's enclosure-side exit.
    pub internal_exit_center_m: [f64; 3],
    /// Unit vector from enclosure to exterior, always `[0, 0, 1]` for the
    /// current planar-front product family.
    pub outward_axis: [f64; 3],
    /// Air-bore radius.
    pub bore_radius_m: f64,
    /// Radius of the explicit solid tube wall.
    pub outer_radius_m: f64,
    /// Physical bore length, before acoustic end correction.
    pub length_m: f64,
}

impl ResolvedPortGeometry {
    /// Cross-sectional air area used for port volume-flow integration.
    pub fn bore_area_m2(self) -> f64 {
        std::f64::consts::PI * self.bore_radius_m.powi(2)
    }

    /// Resolve a schema-level circular port into the shared product frame.
    ///
    /// A missing wall thickness is intentionally rejected: it is sufficient
    /// for the legacy lumped impedance but not for a solid/air PLC.
    pub fn from_design(port: PortDesign) -> Result<Self, ProductInteriorMeshError> {
        let wall_thickness_m = port
            .wall_thickness_m
            .ok_or(ProductInteriorMeshError::MissingPortWallGeometry)?;
        let bore_radius_m = port.diameter_m * 0.5;
        let outer_radius_m = bore_radius_m + wall_thickness_m;
        if !bore_radius_m.is_finite()
            || !outer_radius_m.is_finite()
            || !port.length_m.is_finite()
            || bore_radius_m <= 0.0
            || outer_radius_m <= bore_radius_m
            || port.length_m <= 0.0
        {
            return Err(ProductInteriorMeshError::InvalidMeshConfig(
                "resolved port dimensions must be finite and positive",
            ));
        }
        Ok(Self {
            mouth_center_m: [port.position_m[0], port.position_m[1], 0.0],
            internal_exit_center_m: [port.position_m[0], port.position_m[1], -port.length_m],
            outward_axis: [0.0, 0.0, 1.0],
            bore_radius_m,
            outer_radius_m,
            length_m: port.length_m,
        })
    }
}

/// Axis-aligned solid extent used by the bounded Cartesian brace lowering.
///
/// It is intentionally private: this is an intermediate representation, not
/// a claim that the generated stair-step boundary is body fitted.
#[derive(Debug, Clone, Copy, PartialEq)]
struct AxisAlignedSolid {
    min_m: [f64; 3],
    max_m: [f64; 3],
}

impl AxisAlignedSolid {
    fn contains(self, point: [f64; 3]) -> bool {
        (0..3).all(|axis| point[axis] >= self.min_m[axis] && point[axis] <= self.max_m[axis])
    }

    fn contains_expanded(self, point: [f64; 3], margin_m: f64) -> bool {
        (0..3).all(|axis| {
            point[axis] >= self.min_m[axis] - margin_m && point[axis] <= self.max_m[axis] + margin_m
        })
    }

    fn overlaps_volume(self, other: Self) -> bool {
        (0..3).all(|axis| {
            self.min_m[axis] < other.max_m[axis] && other.min_m[axis] < self.max_m[axis]
        })
    }
}

/// Physical solid extent of one panel rib in the product coordinate frame.
#[derive(Debug, Clone, Copy, PartialEq)]
struct ResolvedBraceGeometry {
    solid: AxisAlignedSolid,
}

impl ResolvedBraceGeometry {
    fn from_design(
        cabinet: &CabinetDesign,
        brace: BraceDesign,
        index: usize,
    ) -> Result<Self, ProductInteriorMeshError> {
        let cross_section = brace
            .cross_section
            .ok_or(ProductInteriorMeshError::MissingBraceCrossSection)?;
        let (w, h, d) = (
            cabinet.box_width_m,
            cabinet.box_height_m,
            cabinet.box_depth_m,
        );
        let (in_plane_axis, normal_axis, in_plane_min, in_plane_length, normal_min, normal_max) =
            match (brace.panel, brace.split_along) {
                (PanelId::Top, PanelAxis::Width) => {
                    (0, 1, -w * 0.5, w, h * 0.5 - cross_section.depth_m, h * 0.5)
                }
                (PanelId::Top, PanelAxis::Height) => {
                    (2, 1, -d, d, h * 0.5 - cross_section.depth_m, h * 0.5)
                }
                (PanelId::Bottom, PanelAxis::Width) => (
                    0,
                    1,
                    -w * 0.5,
                    w,
                    -h * 0.5,
                    -h * 0.5 + cross_section.depth_m,
                ),
                (PanelId::Bottom, PanelAxis::Height) => {
                    (2, 1, -d, d, -h * 0.5, -h * 0.5 + cross_section.depth_m)
                }
                (PanelId::Left, PanelAxis::Width) => {
                    (2, 0, -d, d, -w * 0.5, -w * 0.5 + cross_section.depth_m)
                }
                (PanelId::Left, PanelAxis::Height) => (
                    1,
                    0,
                    -h * 0.5,
                    h,
                    -w * 0.5,
                    -w * 0.5 + cross_section.depth_m,
                ),
                (PanelId::Right, PanelAxis::Width) => {
                    (2, 0, -d, d, w * 0.5 - cross_section.depth_m, w * 0.5)
                }
                (PanelId::Right, PanelAxis::Height) => {
                    (1, 0, -h * 0.5, h, w * 0.5 - cross_section.depth_m, w * 0.5)
                }
                (PanelId::Back, PanelAxis::Width) => {
                    (0, 2, -w * 0.5, w, -d, -d + cross_section.depth_m)
                }
                (PanelId::Back, PanelAxis::Height) => {
                    (1, 2, -h * 0.5, h, -d, -d + cross_section.depth_m)
                }
            };
        let center = in_plane_min + brace.position_fraction * in_plane_length;
        let mut min_m = [-w * 0.5, -h * 0.5, -d];
        let mut max_m = [w * 0.5, h * 0.5, 0.0];
        min_m[in_plane_axis] = center - cross_section.width_m * 0.5;
        max_m[in_plane_axis] = center + cross_section.width_m * 0.5;
        min_m[normal_axis] = normal_min;
        max_m[normal_axis] = normal_max;
        let solid = AxisAlignedSolid { min_m, max_m };
        if !cross_section.width_m.is_finite()
            || !cross_section.depth_m.is_finite()
            || cross_section.width_m <= 0.0
            || cross_section.depth_m <= 0.0
            || (0..3).any(|axis| {
                min_m[axis] < [-w * 0.5, -h * 0.5, -d][axis]
                    || max_m[axis] > [w * 0.5, h * 0.5, 0.0][axis]
            })
        {
            return Err(ProductInteriorMeshError::BraceOutsidePanel { index });
        }
        Ok(Self { solid })
    }
}

/// One of the six rigid walls of the rectangular interior cavity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum InteriorWall {
    Left,
    Right,
    Bottom,
    Top,
    Back,
    Front,
}

impl InteriorWall {
    const ALL: [Self; 6] = [
        Self::Left,
        Self::Right,
        Self::Bottom,
        Self::Top,
        Self::Back,
        Self::Front,
    ];

    const fn marker(self) -> i32 {
        WALL_MARKER_BASE
            + match self {
                Self::Left => 0,
                Self::Right => 1,
                Self::Bottom => 2,
                Self::Top => 3,
                Self::Back => 4,
                Self::Front => 5,
            }
    }

    const fn opposite(self) -> Self {
        match self {
            Self::Left => Self::Right,
            Self::Right => Self::Left,
            Self::Bottom => Self::Top,
            Self::Top => Self::Bottom,
            Self::Back => Self::Front,
            Self::Front => Self::Back,
        }
    }

    /// Outward normal of this cabinet wall in product coordinates.
    const fn outward_normal(self) -> [f64; 3] {
        match self {
            Self::Left => [-1.0, 0.0, 0.0],
            Self::Right => [1.0, 0.0, 0.0],
            Self::Bottom => [0.0, -1.0, 0.0],
            Self::Top => [0.0, 1.0, 0.0],
            Self::Back => [0.0, 0.0, -1.0],
            Self::Front => [0.0, 0.0, 1.0],
        }
    }
}

/// Semantic identity of one product interior boundary patch.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum ProductInteriorPatchKind {
    RigidWall(InteriorWall),
    DriverRear {
        driver_index: usize,
    },
    /// Interior surface of a product-owned body-fitted waveguide.
    WaveguideWall {
        driver_index: usize,
    },
    /// Exterior-facing opening of an explicitly meshed port bore.
    PortMouth,
    /// Interior boundary of the explicit solid port tube.
    PortWall,
    /// Interior surface of one explicitly resolved panel-rib brace.
    BraceWall {
        brace_index: usize,
    },
}

/// A stable, marked boundary patch in the interior FEM mesh.
///
/// A `RigidWall` entry is intentionally absent when a boundary-attached brace
/// fully covers that cabinet wall: the fluid domain then sees the brace's
/// exposed face rather than a coincident duplicate cabinet-wall surface.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProductInteriorPatch {
    pub kind: ProductInteriorPatchKind,
    /// Physical marker written to the FEM boundary faces.
    pub marker: i32,
    /// Boundary-face indices in the returned [`FemMesh`].
    pub boundary_faces: Vec<usize>,
    /// Sum of triangular face areas in square metres.
    pub area_m2: f64,
    /// Canonical outward normal when the patch has one constant direction.
    ///
    /// Composite solid walls such as a port tube or brace intentionally use
    /// `None`; driver patches retain the declared diaphragm normal, including
    /// bounded tilted apertures.
    #[serde(default)]
    pub canonical_normal_m: Option<[f64; 3]>,
}

/// Resolution and resource policy for the sealed product interior mesh.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ProductInteriorMeshConfig {
    /// Largest requested local element edge before tetrahedral subdivision.
    ///
    /// Cartesian lowering applies this on each box axis. Body-fitted lowering
    /// applies it to the circular driver-rim arc length.
    pub max_element_size_m: f64,
    /// Minimum Cartesian cell count on every cavity axis.
    ///
    /// Body-fitted lowering instead retains its fixed circular-area accuracy
    /// floor, independent of this Cartesian grid policy.
    pub min_cells_per_axis: usize,
    /// Hard resource limit on Cartesian cell count on any cavity axis.
    ///
    /// Body-fitted lowering allows four times this count around the rim, one
    /// deterministic run for each rectangular-cabinet side.
    pub max_cells_per_axis: usize,
}

impl Default for ProductInteriorMeshConfig {
    fn default() -> Self {
        Self {
            max_element_size_m: 0.025,
            min_cells_per_axis: 2,
            max_cells_per_axis: 256,
        }
    }
}

impl ProductInteriorMeshConfig {
    fn validate(self) -> Result<(), ProductInteriorMeshError> {
        if !self.max_element_size_m.is_finite() || self.max_element_size_m <= 0.0 {
            return Err(ProductInteriorMeshError::InvalidMeshConfig(
                "max_element_size_m must be finite and positive",
            ));
        }
        if self.min_cells_per_axis == 0 || self.max_cells_per_axis < self.min_cells_per_axis {
            return Err(ProductInteriorMeshError::InvalidMeshConfig(
                "cell-count bounds must be positive and ordered",
            ));
        }
        Ok(())
    }

    fn cells_for(self, length_m: f64) -> Result<usize, ProductInteriorMeshError> {
        let count = (length_m / self.max_element_size_m)
            .ceil()
            .max(self.min_cells_per_axis as f64) as usize;
        if count > self.max_cells_per_axis {
            return Err(ProductInteriorMeshError::ResourceLimit {
                requested: count,
                maximum: self.max_cells_per_axis,
            });
        }
        Ok(count)
    }
}

/// Diagnostics retained with the canonical interior mesh.
#[derive(Debug, Clone, Copy)]
struct WaveguideStation {
    parameter: f64,
    horizontal_radius_m: f64,
    vertical_radius_m: f64,
}

fn resolve_waveguide_stations(
    waveguide: &crate::waveguide_design::ProductWaveguideDesign,
) -> Result<Vec<WaveguideStation>, ProductInteriorMeshError> {
    let mut parameters = waveguide
        .horizontal_spline
        .iter()
        .chain(&waveguide.vertical_spline)
        .map(|point| point[0])
        .collect::<Vec<_>>();
    parameters.sort_by(f64::total_cmp);
    parameters.dedup_by(|left, right| (*left - *right).abs() <= 1.0e-12);
    if parameters.len() < 2 {
        return Err(ProductInteriorMeshError::InvalidMeshConfig(
            "waveguide requires at least two profile stations",
        ));
    }
    Ok(parameters
        .into_iter()
        .map(|parameter| WaveguideStation {
            parameter,
            horizontal_radius_m: interpolate_waveguide_profile(
                &waveguide.horizontal_spline,
                parameter,
            ),
            vertical_radius_m: interpolate_waveguide_profile(&waveguide.vertical_spline, parameter),
        })
        .collect())
}

fn interpolate_waveguide_profile(points: &[[f64; 2]], parameter: f64) -> f64 {
    for pair in points.windows(2) {
        if parameter <= pair[1][0] {
            let span = pair[1][0] - pair[0][0];
            let fraction = if span > 0.0 {
                (parameter - pair[0][0]) / span
            } else {
                0.0
            };
            return pair[0][1] + fraction * (pair[1][1] - pair[0][1]);
        }
    }
    points.last().map(|point| point[1]).unwrap_or(0.0)
}

fn body_fitted_waveguide_point(
    frame: crate::product_design::DriverApertureFrame,
    depth_m: f64,
    station: WaveguideStation,
    angle: f64,
) -> [f64; 3] {
    let center = [
        frame.wall_center_m[0] - station.parameter * depth_m * frame.wall_normal_m[0],
        frame.wall_center_m[1] - station.parameter * depth_m * frame.wall_normal_m[1],
        frame.wall_center_m[2] - station.parameter * depth_m * frame.wall_normal_m[2],
    ];
    let cosine = angle.cos();
    let sine = angle.sin();
    [
        center[0]
            + station.horizontal_radius_m * cosine * frame.tangent_u_m[0]
            + station.vertical_radius_m * sine * frame.tangent_v_m[0],
        center[1]
            + station.horizontal_radius_m * cosine * frame.tangent_u_m[1]
            + station.vertical_radius_m * sine * frame.tangent_v_m[1],
        center[2]
            + station.horizontal_radius_m * cosine * frame.tangent_u_m[2]
            + station.vertical_radius_m * sine * frame.tangent_v_m[2],
    ]
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProductInteriorMeshDiagnostics {
    pub node_count: usize,
    pub tetrahedron_count: usize,
    pub boundary_face_count: usize,
    pub fluid_volume_m3: f64,
    pub expected_box_volume_m3: f64,
    pub patch_area_m2: BTreeMap<String, f64>,
}

/// Canonical sealed product interior mesh and typed patch registry.
#[derive(Debug, Clone)]
pub struct ProductInteriorMesh {
    pub mesh: FemMesh,
    pub patches: Vec<ProductInteriorPatch>,
    pub diagnostics: ProductInteriorMeshDiagnostics,
    /// Stable geometry-and-meshing identity for cache and evidence checks.
    pub geometry_fingerprint: String,
}

impl ProductInteriorMesh {
    pub fn patch(&self, kind: ProductInteriorPatchKind) -> Option<&ProductInteriorPatch> {
        self.patches.iter().find(|patch| patch.kind == kind)
    }
}

/// Fail-closed errors while lowering a product into the currently supported
/// interior-mesh family.
#[derive(Debug, thiserror::Error)]
pub enum ProductInteriorMeshError {
    #[error(transparent)]
    InvalidProduct(#[from] ProductDesignError),
    #[error("invalid product interior mesh configuration: {0}")]
    InvalidMeshConfig(&'static str),
    #[error("mesh requires {requested} cells on one axis, exceeding policy maximum {maximum}")]
    ResourceLimit { requested: usize, maximum: usize },
    #[error("resolved port geometry is not implemented by the sealed interior mesh")]
    ResolvedPortUnsupported,
    #[error("resolved port mesh requires an explicit tube-wall thickness")]
    MissingPortWallGeometry,
    #[error("port tube wall is not resolved at the requested mesh resolution")]
    PortWallUnresolved,
    #[error("port mouth has no boundary faces at the requested mesh resolution")]
    PortMouthEmpty,
    #[error("port-wall carving produced disconnected interior air domains")]
    DisconnectedFluidDomain,
    #[error("port tube wall intersects brace {brace_index}")]
    PortBraceIntersection { brace_index: usize },
    /// Retained for callers that matched the pre-composition capability gate.
    /// Current port/brace lowering no longer emits this error.
    #[error("resolved brace geometry is not implemented by the sealed interior mesh")]
    ResolvedBraceUnsupported,
    #[error("resolved brace mesh requires an explicit brace cross-section")]
    MissingBraceCrossSection,
    #[error("brace {index} does not fit inside its owning panel")]
    BraceOutsidePanel { index: usize },
    #[error("brace {index} is not resolved at the requested mesh resolution")]
    BraceUnresolved { index: usize },
    #[error("brace solids {first} and {second} overlap")]
    BraceSolidIntersection { first: usize, second: usize },
    #[error("driver {index} is not on a supported rectangular cabinet wall")]
    DriverNotOnCabinetWall { index: usize },
    #[error("driver {index} does not fit in the interior cabinet wall")]
    DriverOutsideInterior { index: usize },
    #[error("driver {index} has no boundary faces at this mesh resolution")]
    DriverPatchEmpty { index: usize },
    #[error("generated mesh lost required patch {kind:?}")]
    MissingPatch { kind: ProductInteriorPatchKind },
    #[error("generated mesh has an invalid boundary face {face}")]
    InvalidBoundaryFace { face: usize },
    #[error("body-fitted sealed lowering does not support this driver/layout combination")]
    BodyFittedDriverLayoutUnsupported,
    #[error(
        "body-fitted driver rim requires {requested} segments, exceeding configured maximum {maximum}"
    )]
    BodyFittedResolutionLimit { requested: usize, maximum: usize },
    #[error("body-fitted waveguide requires one axis-aligned driver on a cabinet face")]
    WaveguideLayoutUnsupported,
}

/// Build a body-fitted sealed-cabinet interior mesh for one bounded
/// cabinet-wall driver, including a bounded tilt from that wall.
///
/// Unlike [`build_sealed_product_interior_mesh`], this path puts the driver
/// rim on an explicit circular surface ring before tetrahedralization. The
/// result therefore preserves a geometric driver boundary rather than a
/// Cartesian centroid classification. Products with ports, braces, or
/// multiple drivers are delegated to the multiwall body-fitted PLC below;
/// that broader path remains axis-aligned and preserves the same typed
/// boundary contract.
pub fn build_body_fitted_sealed_product_interior_mesh(
    design: &ProductDesign,
    config: ProductInteriorMeshConfig,
) -> Result<ProductInteriorMesh, ProductInteriorMeshError> {
    design.validate()?;
    config.validate()?;
    if design.waveguide.is_some()
        && (design.drivers.len() > 1
            || design.cabinet.port.is_some()
            || !design.cabinet.braces.is_empty())
    {
        return Err(ProductInteriorMeshError::WaveguideLayoutUnsupported);
    }
    if design.drivers.len() > 1
        || design.cabinet.port.is_some()
        || !design.cabinet.braces.is_empty()
    {
        return build_multiwall_body_fitted_sealed_product_interior_mesh(design, config);
    }
    let [driver] = design.drivers.as_slice() else {
        return Err(ProductInteriorMeshError::BodyFittedDriverLayoutUnsupported);
    };
    let waveguide = design.waveguide.as_ref();
    if let Some(waveguide) = waveguide
        && (!driver.is_axis_aligned() || waveguide.driver_id != driver.id)
    {
        return Err(ProductInteriorMeshError::WaveguideLayoutUnsupported);
    }
    let waveguide_stations = waveguide.map(resolve_waveguide_stations).transpose()?;
    let cabinet = &design.cabinet;
    let frame = driver.aperture_frame(
        cabinet.baffle_width_m,
        cabinet.baffle_height_m,
        cabinet.box_width_m,
        cabinet.box_height_m,
        cabinet.box_depth_m,
    )?;
    let wall = body_fitted_wall_for_face(frame.mounting_face);
    let oriented = !driver.is_axis_aligned();
    let maximum_ring_radius_m = waveguide_stations
        .as_deref()
        .and_then(|stations| {
            stations
                .iter()
                .map(|station| station.horizontal_radius_m.max(station.vertical_radius_m))
                .max_by(f64::total_cmp)
        })
        .unwrap_or(driver.radius_m);
    let ring_segments = body_fitted_ring_segments(maximum_ring_radius_m, config)?;
    let driver_marker = marker_for(ProductInteriorPatchKind::DriverRear { driver_index: 0 })?;
    let (panel_boundary, side_markers) = body_fitted_panel_boundary(cabinet, wall, ring_segments);
    let center = body_fitted_panel_coordinates(cabinet, wall, driver.position_m);
    let (panel_width, panel_height) = body_fitted_panel_dimensions(cabinet, wall);
    let panel_radius_u_m = waveguide_stations
        .as_deref()
        .and_then(|stations| stations.first())
        .map(|station| station.horizontal_radius_m)
        .unwrap_or(driver.radius_m);
    let panel_radius_v_m = waveguide_stations
        .as_deref()
        .and_then(|stations| stations.first())
        .map(|station| station.vertical_radius_m)
        .unwrap_or(driver.radius_m);
    if center[0].abs() + panel_radius_u_m >= panel_width * 0.5
        || center[1].abs() + panel_radius_v_m >= panel_height * 0.5
    {
        return Err(ProductInteriorMeshError::BodyFittedDriverLayoutUnsupported);
    }
    if let Some(waveguide) = waveguide {
        let cavity_depth_m = match wall {
            InteriorWall::Front | InteriorWall::Back => cabinet.box_depth_m,
            InteriorWall::Left | InteriorWall::Right => cabinet.box_width_m,
            InteriorWall::Top | InteriorWall::Bottom => cabinet.box_height_m,
        };
        if waveguide.depth_m >= cavity_depth_m {
            return Err(ProductInteriorMeshError::WaveguideLayoutUnsupported);
        }
    }
    let center_point = body_fitted_panel_point(cabinet, wall, center);
    let driver_center_point = if let Some(waveguide) = waveguide {
        [
            frame.wall_center_m[0] - waveguide.depth_m * frame.wall_normal_m[0],
            frame.wall_center_m[1] - waveguide.depth_m * frame.wall_normal_m[1],
            frame.wall_center_m[2] - waveguide.depth_m * frame.wall_normal_m[2],
        ]
    } else if oriented {
        frame.center_m
    } else {
        center_point
    };
    let inward_offset = body_fitted_inward_offset(cabinet, wall);
    let mut nodes = vec![Point::new_3d(
        driver_center_point[0],
        driver_center_point[1],
        driver_center_point[2],
    )];
    let panel_center = 0;
    let mut driver_ring = Vec::with_capacity(ring_segments);
    let mut panel_ring = Vec::with_capacity(ring_segments);
    let mut outer_ring = Vec::with_capacity(ring_segments);
    let first_outer = panel_boundary[0];
    let first_angle = (first_outer[1] - center[1]).atan2(first_outer[0] - center[0]);
    for (index, &point) in panel_boundary.iter().enumerate() {
        let angle = if oriented {
            std::f64::consts::TAU * index as f64 / ring_segments as f64
        } else {
            first_angle + std::f64::consts::TAU * index as f64 / ring_segments as f64
        };
        let inner = if oriented {
            frame.aperture_point(angle)
        } else {
            body_fitted_panel_point(
                cabinet,
                wall,
                [
                    center[0] + driver.radius_m * angle.cos(),
                    center[1] + driver.radius_m * angle.sin(),
                ],
            )
        };
        let panel = if oriented {
            frame.project_to_wall(inner)
        } else {
            inner
        };
        let (driver_index, panel_index) = if waveguide.is_some() {
            (panel_center, panel_center)
        } else {
            let driver_index = nodes.len();
            nodes.push(Point::new_3d(inner[0], inner[1], inner[2]));
            let panel_index = if oriented {
                let index = nodes.len();
                nodes.push(Point::new_3d(panel[0], panel[1], panel[2]));
                index
            } else {
                driver_index
            };
            (driver_index, panel_index)
        };
        driver_ring.push(driver_index);
        panel_ring.push(panel_index);
        let outer = body_fitted_panel_point(cabinet, wall, point);
        outer_ring.push(nodes.len());
        nodes.push(Point::new_3d(outer[0], outer[1], outer[2]));
    }
    let mut waveguide_rings = Vec::new();
    if let (Some(waveguide), Some(stations)) = (waveguide, waveguide_stations.as_deref()) {
        for station in stations {
            let ring = (0..ring_segments)
                .map(|index| {
                    let angle = std::f64::consts::TAU * index as f64 / ring_segments as f64;
                    let point =
                        body_fitted_waveguide_point(frame, waveguide.depth_m, *station, angle);
                    let node = nodes.len();
                    nodes.push(Point::new_3d(point[0], point[1], point[2]));
                    node
                })
                .collect::<Vec<_>>();
            waveguide_rings.push(ring);
        }
        panel_ring = waveguide_rings
            .first()
            .cloned()
            .expect("validated waveguide has a mouth ring");
        driver_ring = waveguide_rings
            .last()
            .cloned()
            .expect("validated waveguide has a throat ring");
    }

    let opposite_center = nodes.len();
    nodes.push(Point::new_3d(
        center_point[0] + inward_offset[0],
        center_point[1] + inward_offset[1],
        center_point[2] + inward_offset[2],
    ));
    let opposite_ring = panel_boundary
        .iter()
        .map(|&point| {
            let point = body_fitted_panel_point(cabinet, wall, point);
            let index = nodes.len();
            nodes.push(Point::new_3d(
                point[0] + inward_offset[0],
                point[1] + inward_offset[1],
                point[2] + inward_offset[2],
            ));
            index
        })
        .collect::<Vec<_>>();

    let mut facets = Vec::with_capacity(ring_segments * 6);
    for index in 0..ring_segments {
        let next = (index + 1) % ring_segments;
        // The panel disk and annulus together tile the full baffle plane, with
        // the driver marker exactly bounded by the circular ring.
        facets.push((
            [panel_center, driver_ring[index], driver_ring[next]],
            driver_marker,
        ));
        let panel_marker = marker_for(ProductInteriorPatchKind::RigidWall(wall))?;
        facets.push((
            [panel_ring[index], outer_ring[index], outer_ring[next]],
            panel_marker,
        ));
        facets.push((
            [panel_ring[index], outer_ring[next], panel_ring[next]],
            panel_marker,
        ));
        if waveguide_rings.len() > 1 {
            let waveguide_marker =
                marker_for(ProductInteriorPatchKind::WaveguideWall { driver_index: 0 })?;
            for pair in waveguide_rings.windows(2) {
                facets.push((
                    [pair[0][index], pair[1][index], pair[1][next]],
                    waveguide_marker,
                ));
                facets.push((
                    [pair[0][index], pair[1][next], pair[0][next]],
                    waveguide_marker,
                ));
            }
        }
        if oriented {
            facets.push((
                [panel_ring[index], driver_ring[next], driver_ring[index]],
                panel_marker,
            ));
            facets.push((
                [panel_ring[index], panel_ring[next], driver_ring[next]],
                panel_marker,
            ));
        }
        // Reverse the opposing face winding so its normal points outward.
        facets.push((
            [opposite_center, opposite_ring[next], opposite_ring[index]],
            marker_for(ProductInteriorPatchKind::RigidWall(wall.opposite()))?,
        ));
        let side_marker = marker_for(ProductInteriorPatchKind::RigidWall(side_markers[index]))?;
        facets.push((
            [outer_ring[index], opposite_ring[index], opposite_ring[next]],
            side_marker,
        ));
        facets.push((
            [outer_ring[index], opposite_ring[next], outer_ring[next]],
            side_marker,
        ));
    }

    if oriented {
        body_fitted_orient_surface_facets(&nodes, &mut facets);
    }

    // The enclosure is convex, so a deterministic fan from one interior node
    // to each explicitly triangulated surface facet is a non-overlapping,
    // body-fitted tetrahedralization. Keeping this direct construction avoids
    // sending the coplanar baffle tiling through a general recovery pass that
    // is deliberately bounded and may reject it after its Steiner budget.
    let interior = nodes.len();
    nodes.push(Point::new_3d(0.0, 0.0, -cabinet.box_depth_m * 0.5));
    let mut mesh = FemMesh::new(3);
    for node in nodes {
        mesh.add_node(node);
    }
    for &(facet, _) in &facets {
        mesh.add_element(
            ElementType::Tetrahedron,
            vec![interior, facet[0], facet[1], facet[2]],
        );
    }
    mesh.detect_boundaries();
    for boundary in &mut mesh.boundaries {
        let marker = facets
            .iter()
            .find_map(|(facet, marker)| {
                same_triangle_nodes(&boundary.nodes, facet).then_some(*marker)
            })
            .ok_or(ProductInteriorMeshError::InvalidBoundaryFace {
                face: boundary.element_idx,
            })?;
        boundary.boundary_type = BoundaryType::Neumann;
        boundary.marker = marker;
    }
    body_fitted_product_mesh_result(design, config, ring_segments, mesh)
}

/// Body-fitted closed-surface fan for any number of non-overlapping drivers
/// on axis-aligned rectangular cabinet walls.  Each wall panel is a convex
/// polygon with zero or more circular holes, triangulated as a constrained
/// planar PLC; every driver disk is then added as its own marked facet fan.
fn build_multiwall_body_fitted_sealed_product_interior_mesh(
    design: &ProductDesign,
    config: ProductInteriorMeshConfig,
) -> Result<ProductInteriorMesh, ProductInteriorMeshError> {
    let cabinet = &design.cabinet;
    let port = design
        .cabinet
        .port
        .map(ResolvedPortGeometry::from_design)
        .transpose()?;
    let braces = design
        .cabinet
        .braces
        .iter()
        .copied()
        .enumerate()
        .map(|(index, brace)| ResolvedBraceGeometry::from_design(cabinet, brace, index))
        .collect::<Result<Vec<_>, _>>()?;
    if let Some(port) = port {
        for (brace_index, brace) in braces.iter().enumerate() {
            if brace_intersects_port(brace.solid, port) {
                return Err(ProductInteriorMeshError::PortBraceIntersection { brace_index });
            }
        }
    }
    let driver_frames = design
        .drivers
        .iter()
        .map(|driver| {
            driver.aperture_frame(
                cabinet.baffle_width_m,
                cabinet.baffle_height_m,
                cabinet.box_width_m,
                cabinet.box_height_m,
                cabinet.box_depth_m,
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    let max_radius = driver_frames
        .iter()
        .map(|frame| frame.projected_radius_m)
        .reduce(f64::max)
        .into_iter()
        .chain(port.map(|port| port.outer_radius_m))
        .reduce(f64::max)
        .ok_or(ProductInteriorMeshError::BodyFittedDriverLayoutUnsupported)?;
    let ring_segments = body_fitted_ring_segments(max_radius, config)?;
    let mut wall_drivers = std::array::from_fn::<_, 6, _>(|_| Vec::new());
    for (index, (driver, frame)) in design.drivers.iter().zip(&driver_frames).enumerate() {
        let wall = body_fitted_wall_for_face(frame.mounting_face);
        let slot = body_fitted_wall_slot(wall);
        let center = body_fitted_panel_coordinates(cabinet, wall, driver.position_m);
        let (width, height) = body_fitted_panel_dimensions(cabinet, wall);
        if center[0].abs() + frame.projected_radius_m >= width * 0.5
            || center[1].abs() + frame.projected_radius_m >= height * 0.5
        {
            return Err(ProductInteriorMeshError::BodyFittedDriverLayoutUnsupported);
        }
        wall_drivers[slot].push((index, driver, *frame));
    }

    let mut nodes = Vec::new();
    let mut facets = Vec::new();
    for wall in InteriorWall::ALL {
        let (ring_boundary, _) = body_fitted_panel_boundary(cabinet, wall, ring_segments);
        let panel_boundary = [
            ring_boundary[0],
            ring_boundary[ring_segments / 4],
            ring_boundary[ring_segments / 2],
            ring_boundary[ring_segments * 3 / 4],
        ];
        let panel_marker = marker_for(ProductInteriorPatchKind::RigidWall(wall))?;
        let drivers = &wall_drivers[body_fitted_wall_slot(wall)];
        let outer = panel_boundary
            .iter()
            .map(|&[u, v]| (u, v))
            .collect::<Vec<_>>();
        let driver_holes = drivers
            .iter()
            .map(|(_, driver, frame)| {
                if driver.is_axis_aligned() {
                    Ok(body_fitted_driver_ring(
                        cabinet,
                        wall,
                        driver,
                        ring_boundary[0],
                        ring_segments,
                    ))
                } else {
                    Ok((0..ring_segments)
                        .map(|index| {
                            let angle = std::f64::consts::TAU * index as f64 / ring_segments as f64;
                            let point = frame.project_to_wall(frame.aperture_point(angle));
                            let coordinates = body_fitted_panel_coordinates(cabinet, wall, point);
                            (coordinates[0], coordinates[1])
                        })
                        .collect::<Vec<_>>())
                }
            })
            .collect::<Result<Vec<_>, ProductInteriorMeshError>>()?;
        let port_hole = if let (InteriorWall::Front, Some(port)) = (wall, port) {
            if port.mouth_center_m[0].abs() + port.outer_radius_m >= cabinet.box_width_m * 0.5
                || port.mouth_center_m[1].abs() + port.outer_radius_m >= cabinet.box_height_m * 0.5
            {
                return Err(ProductInteriorMeshError::BodyFittedDriverLayoutUnsupported);
            }
            Some(body_fitted_circle_ring(
                [port.mouth_center_m[0], port.mouth_center_m[1]],
                port.outer_radius_m,
                ring_boundary[0],
                ring_segments,
            ))
        } else {
            None
        };
        let mut panel_holes = driver_holes.clone();
        if let Some(port_hole) = &port_hole {
            panel_holes.push(port_hole.clone());
        }
        let brace_footprints = braces
            .iter()
            .filter(|brace| body_fitted_brace_touches_wall(brace.solid, cabinet, wall))
            .map(|brace| body_fitted_brace_footprint(brace.solid, cabinet, wall))
            .collect::<Vec<_>>();
        let panel_regions =
            triangulate_body_fitted_panel_regions(&outer, &brace_footprints, &panel_holes)
                .map_err(|_| ProductInteriorMeshError::BodyFittedDriverLayoutUnsupported)?;
        for (panel_vertices, panel_triangles) in panel_regions {
            let panel_nodes = panel_vertices
                .iter()
                .map(|&point| {
                    body_fitted_insert_node(
                        &mut nodes,
                        body_fitted_panel_point(cabinet, wall, [point.0, point.1]),
                    )
                })
                .collect::<Vec<_>>();
            for triangle in panel_triangles {
                body_fitted_push_outward_facet(
                    &mut facets,
                    &nodes,
                    triangle.map(|index| panel_nodes[index]),
                    panel_marker,
                    wall,
                );
            }
        }
        for ((driver_index, driver, frame), ring) in drivers.iter().zip(&driver_holes) {
            let oriented = !driver.is_axis_aligned();
            let center = if oriented {
                frame.center_m
            } else {
                let center = body_fitted_panel_coordinates(cabinet, wall, driver.position_m);
                body_fitted_panel_point(cabinet, wall, center)
            };
            let center = body_fitted_insert_node(&mut nodes, center);
            let panel_ring = ring
                .iter()
                .map(|&point| {
                    body_fitted_insert_node(
                        &mut nodes,
                        body_fitted_panel_point(cabinet, wall, [point.0, point.1]),
                    )
                })
                .collect::<Vec<_>>();
            let driver_ring = if oriented {
                (0..ring_segments)
                    .map(|index| {
                        let angle = std::f64::consts::TAU * index as f64 / ring_segments as f64;
                        body_fitted_insert_node(&mut nodes, frame.aperture_point(angle))
                    })
                    .collect::<Vec<_>>()
            } else {
                panel_ring.clone()
            };
            let throat_ring = if oriented {
                Some(
                    (0..ring_segments)
                        .map(|index| {
                            let angle = std::f64::consts::TAU * index as f64 / ring_segments as f64;
                            let wall_point = body_fitted_panel_point(
                                cabinet,
                                wall,
                                [ring[index].0, ring[index].1],
                            );
                            let aperture_point = frame.aperture_point(angle);
                            body_fitted_insert_node(
                                &mut nodes,
                                [
                                    0.5 * (wall_point[0] + aperture_point[0]),
                                    0.5 * (wall_point[1] + aperture_point[1]),
                                    0.5 * (wall_point[2] + aperture_point[2]),
                                ],
                            )
                        })
                        .collect::<Vec<_>>(),
                )
            } else {
                None
            };
            let driver_marker = marker_for(ProductInteriorPatchKind::DriverRear {
                driver_index: *driver_index,
            })?;
            for index in 0..ring_segments {
                let next = (index + 1) % ring_segments;
                if oriented {
                    facets.push((
                        [center, driver_ring[index], driver_ring[next]],
                        driver_marker,
                    ));
                    // The annular throat is cabinet structure, not a driven
                    // diaphragm. Keep it on the rigid-wall marker so the
                    // monolithic bridge only couples the actual disk.
                    facets.push((
                        [
                            panel_ring[index],
                            throat_ring.as_ref().expect("oriented throat ring")[next],
                            throat_ring.as_ref().expect("oriented throat ring")[index],
                        ],
                        panel_marker,
                    ));
                    facets.push((
                        [
                            panel_ring[index],
                            panel_ring[next],
                            throat_ring.as_ref().expect("oriented throat ring")[next],
                        ],
                        panel_marker,
                    ));
                    facets.push((
                        [
                            throat_ring.as_ref().expect("oriented throat ring")[index],
                            throat_ring.as_ref().expect("oriented throat ring")[next],
                            driver_ring[next],
                        ],
                        panel_marker,
                    ));
                    facets.push((
                        [
                            throat_ring.as_ref().expect("oriented throat ring")[index],
                            driver_ring[next],
                            driver_ring[index],
                        ],
                        panel_marker,
                    ));
                } else {
                    body_fitted_push_outward_facet(
                        &mut facets,
                        &nodes,
                        [center, driver_ring[index], driver_ring[next]],
                        driver_marker,
                        wall,
                    );
                }
            }
        }
        if let (InteriorWall::Front, Some(port)) = (wall, port) {
            let outer = body_fitted_circle_ring(
                [port.mouth_center_m[0], port.mouth_center_m[1]],
                port.outer_radius_m,
                ring_boundary[0],
                ring_segments,
            );
            let inner = body_fitted_circle_ring(
                [port.mouth_center_m[0], port.mouth_center_m[1]],
                port.bore_radius_m,
                ring_boundary[0],
                ring_segments,
            );
            let outer_front = outer
                .iter()
                .map(|&point| body_fitted_insert_node(&mut nodes, [point.0, point.1, 0.0]))
                .collect::<Vec<_>>();
            let inner_front = inner
                .iter()
                .map(|&point| body_fitted_insert_node(&mut nodes, [point.0, point.1, 0.0]))
                .collect::<Vec<_>>();
            let outer_back = outer
                .iter()
                .map(|&point| {
                    body_fitted_insert_node(&mut nodes, [point.0, point.1, -port.length_m])
                })
                .collect::<Vec<_>>();
            let inner_back = inner
                .iter()
                .map(|&point| {
                    body_fitted_insert_node(&mut nodes, [point.0, point.1, -port.length_m])
                })
                .collect::<Vec<_>>();
            let outer_mid = outer
                .iter()
                .map(|&point| {
                    body_fitted_insert_node(&mut nodes, [point.0, point.1, -port.length_m * 0.5])
                })
                .collect::<Vec<_>>();
            let inner_mid = inner
                .iter()
                .map(|&point| {
                    body_fitted_insert_node(&mut nodes, [point.0, point.1, -port.length_m * 0.5])
                })
                .collect::<Vec<_>>();
            let mouth_center = body_fitted_insert_node(
                &mut nodes,
                [port.mouth_center_m[0], port.mouth_center_m[1], 0.0],
            );
            let wall_marker = marker_for(ProductInteriorPatchKind::PortWall)?;
            let mouth_marker = marker_for(ProductInteriorPatchKind::PortMouth)?;
            for index in 0..ring_segments {
                let next = (index + 1) % ring_segments;
                body_fitted_push_outward_facet(
                    &mut facets,
                    &nodes,
                    [mouth_center, inner_front[index], inner_front[next]],
                    mouth_marker,
                    wall,
                );
                facets.push((
                    [outer_front[index], outer_mid[index], outer_mid[next]],
                    wall_marker,
                ));
                facets.push((
                    [outer_front[index], outer_mid[next], outer_front[next]],
                    wall_marker,
                ));
                facets.push((
                    [inner_front[index], inner_mid[next], inner_mid[index]],
                    wall_marker,
                ));
                facets.push((
                    [inner_front[index], inner_front[next], inner_mid[next]],
                    wall_marker,
                ));
                facets.push((
                    [outer_mid[index], outer_back[index], outer_back[next]],
                    wall_marker,
                ));
                facets.push((
                    [outer_mid[index], outer_back[next], outer_mid[next]],
                    wall_marker,
                ));
                facets.push((
                    [inner_mid[index], inner_back[next], inner_back[index]],
                    wall_marker,
                ));
                facets.push((
                    [inner_mid[index], inner_mid[next], inner_back[next]],
                    wall_marker,
                ));
                facets.push((
                    [outer_back[index], inner_back[next], inner_back[index]],
                    wall_marker,
                ));
                facets.push((
                    [outer_back[index], outer_back[next], inner_back[next]],
                    wall_marker,
                ));
            }
        }
        for (brace_index, brace) in braces.iter().enumerate() {
            if body_fitted_brace_wall(brace.solid, cabinet) == Some(wall) {
                let marker = marker_for(ProductInteriorPatchKind::BraceWall { brace_index })?;
                body_fitted_push_brace_faces(
                    &mut nodes,
                    &mut facets,
                    brace.solid,
                    cabinet,
                    wall,
                    marker,
                );
            }
        }
    }

    let mut facets = body_fitted_refine_surface_facets(&nodes, &facets);
    body_fitted_orient_surface_facets(&nodes, &mut facets);
    let mut mesh = if port.is_some() || !braces.is_empty() {
        tetrahedralize_surface_with_facets(
            &nodes,
            &facets
                .iter()
                .map(|(nodes, marker)| SurfaceFacet {
                    nodes: *nodes,
                    marker: Some(*marker),
                })
                .collect::<Vec<_>>(),
            &TetrahedralizeConfig {
                max_smooth_iterations: 0,
                add_interior_points: false,
                lloyd_iterations: 0,
                preserve_surface: true,
                max_ratio: 1.5,
            },
        )
        .map_err(|_| ProductInteriorMeshError::BodyFittedDriverLayoutUnsupported)?
    } else {
        let interior = body_fitted_insert_node(&mut nodes, [0.0, 0.0, -cabinet.box_depth_m * 0.5]);
        let mut mesh = FemMesh::new(3);
        for &node in &nodes {
            mesh.add_node(node);
        }
        for &(facet, _) in &facets {
            mesh.add_element(
                ElementType::Tetrahedron,
                vec![interior, facet[0], facet[1], facet[2]],
            );
        }
        mesh.detect_boundaries();
        mesh
    };
    for boundary in &mut mesh.boundaries {
        // The constrained tetrahedralizer may retriangulate coplanar input
        // facets (for example, by choosing the opposite diagonal of a wall
        // quad).  It propagates the recovered facet marker onto the generated
        // boundary face, so matching the original node triplet here would
        // reject a valid boundary face.  Product markers are all positive;
        // retain the error for an actually unmarked face.
        let marker = if boundary.marker != 0 {
            boundary.marker
        } else {
            facets
                .iter()
                .find_map(|(facet, marker)| {
                    same_triangle_nodes(&boundary.nodes, facet).then_some(*marker)
                })
                .ok_or(ProductInteriorMeshError::InvalidBoundaryFace {
                    face: boundary.element_idx,
                })?
        };
        boundary.boundary_type = BoundaryType::Neumann;
        boundary.marker = marker;
    }
    body_fitted_product_mesh_result(design, config, ring_segments, mesh)
}

fn body_fitted_wall_slot(wall: InteriorWall) -> usize {
    match wall {
        InteriorWall::Left => 0,
        InteriorWall::Right => 1,
        InteriorWall::Bottom => 2,
        InteriorWall::Top => 3,
        InteriorWall::Back => 4,
        InteriorWall::Front => 5,
    }
}

fn body_fitted_insert_node(nodes: &mut Vec<Point>, point: [f64; 3]) -> usize {
    if let Some((index, _)) = nodes.iter().enumerate().find(|(_, node)| {
        (node.x - point[0]).abs() <= 1.0e-12
            && (node.y - point[1]).abs() <= 1.0e-12
            && (node.z - point[2]).abs() <= 1.0e-12
    }) {
        index
    } else {
        let index = nodes.len();
        nodes.push(Point::new_3d(point[0], point[1], point[2]));
        index
    }
}

fn body_fitted_push_outward_facet(
    facets: &mut Vec<([usize; 3], i32)>,
    nodes: &[Point],
    mut facet: [usize; 3],
    marker: i32,
    wall: InteriorWall,
) {
    let a = &nodes[facet[0]];
    let b = &nodes[facet[1]];
    let c = &nodes[facet[2]];
    let cross = [
        (b.y - a.y) * (c.z - a.z) - (b.z - a.z) * (c.y - a.y),
        (b.z - a.z) * (c.x - a.x) - (b.x - a.x) * (c.z - a.z),
        (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x),
    ];
    let normal = match wall {
        InteriorWall::Left => [-1.0, 0.0, 0.0],
        InteriorWall::Right => [1.0, 0.0, 0.0],
        InteriorWall::Bottom => [0.0, -1.0, 0.0],
        InteriorWall::Top => [0.0, 1.0, 0.0],
        InteriorWall::Back => [0.0, 0.0, -1.0],
        InteriorWall::Front => [0.0, 0.0, 1.0],
    };
    if cross[0] * normal[0] + cross[1] * normal[1] + cross[2] * normal[2] < 0.0 {
        facet.swap(1, 2);
    }
    facets.push((facet, marker));
}

fn body_fitted_driver_ring(
    cabinet: &CabinetDesign,
    wall: InteriorWall,
    driver: &crate::product_design::ProductDriver,
    first_panel_point: [f64; 2],
    ring_segments: usize,
) -> Vec<(f64, f64)> {
    let center = body_fitted_panel_coordinates(cabinet, wall, driver.position_m);
    body_fitted_circle_ring(center, driver.radius_m, first_panel_point, ring_segments)
}

fn body_fitted_circle_ring(
    center: [f64; 2],
    radius_m: f64,
    first_panel_point: [f64; 2],
    ring_segments: usize,
) -> Vec<(f64, f64)> {
    let first = first_panel_point;
    let first_angle = (first[1] - center[1]).atan2(first[0] - center[0]);
    (0..ring_segments)
        .map(|index| {
            let angle = first_angle + std::f64::consts::TAU * index as f64 / ring_segments as f64;
            (
                center[0] + radius_m * angle.cos(),
                center[1] + radius_m * angle.sin(),
            )
        })
        .collect()
}

fn body_fitted_brace_touches_wall(
    solid: AxisAlignedSolid,
    cabinet: &CabinetDesign,
    wall: InteriorWall,
) -> bool {
    const TOLERANCE: f64 = 1.0e-12;
    match wall {
        InteriorWall::Left => (solid.min_m[0] + cabinet.box_width_m * 0.5).abs() <= TOLERANCE,
        InteriorWall::Right => (solid.max_m[0] - cabinet.box_width_m * 0.5).abs() <= TOLERANCE,
        InteriorWall::Bottom => (solid.min_m[1] + cabinet.box_height_m * 0.5).abs() <= TOLERANCE,
        InteriorWall::Top => (solid.max_m[1] - cabinet.box_height_m * 0.5).abs() <= TOLERANCE,
        InteriorWall::Back => (solid.min_m[2] + cabinet.box_depth_m).abs() <= TOLERANCE,
        InteriorWall::Front => solid.max_m[2].abs() <= TOLERANCE,
    }
}

fn body_fitted_brace_wall(
    solid: AxisAlignedSolid,
    cabinet: &CabinetDesign,
) -> Option<InteriorWall> {
    let tolerance = 1.0e-12;
    let extents = [
        solid.max_m[0] - solid.min_m[0],
        solid.max_m[1] - solid.min_m[1],
        solid.max_m[2] - solid.min_m[2],
    ];
    let mut candidates = Vec::with_capacity(3);
    if (solid.max_m[1] - cabinet.box_height_m * 0.5).abs() <= tolerance {
        candidates.push((extents[1], 0, InteriorWall::Top));
    }
    if (solid.min_m[1] + cabinet.box_height_m * 0.5).abs() <= tolerance {
        candidates.push((extents[1], 1, InteriorWall::Bottom));
    }
    if (solid.min_m[0] + cabinet.box_width_m * 0.5).abs() <= tolerance {
        candidates.push((extents[0], 2, InteriorWall::Left));
    }
    if (solid.max_m[0] - cabinet.box_width_m * 0.5).abs() <= tolerance {
        candidates.push((extents[0], 3, InteriorWall::Right));
    }
    if (solid.min_m[2] + cabinet.box_depth_m).abs() <= tolerance {
        candidates.push((extents[2], 4, InteriorWall::Back));
    }
    candidates
        .into_iter()
        .min_by(|left, right| left.0.total_cmp(&right.0).then(left.1.cmp(&right.1)))
        .map(|(_, _, wall)| wall)
}

fn body_fitted_brace_footprint(
    solid: AxisAlignedSolid,
    cabinet: &CabinetDesign,
    wall: InteriorWall,
) -> Vec<(f64, f64)> {
    body_fitted_brace_contact_vertices(solid, wall)
        .into_iter()
        .map(|point| {
            let [u, v] = body_fitted_panel_coordinates(cabinet, wall, point);
            (u, v)
        })
        .collect()
}

fn body_fitted_wall_is_covered_by_brace(
    design: &ProductDesign,
    wall: InteriorWall,
) -> Result<bool, ProductInteriorMeshError> {
    let (outer, _) = body_fitted_panel_boundary(&design.cabinet, wall, 4);
    let outer = outer.iter().map(|&[u, v]| (u, v)).collect::<Vec<_>>();
    let (outer_min_u, outer_max_u, outer_min_v, outer_max_v) = polygon_bounds(&outer)
        .ok_or(ProductInteriorMeshError::BodyFittedDriverLayoutUnsupported)?;
    for (index, brace) in design.cabinet.braces.iter().copied().enumerate() {
        let resolved = ResolvedBraceGeometry::from_design(&design.cabinet, brace, index)?;
        if !body_fitted_brace_touches_wall(resolved.solid, &design.cabinet, wall) {
            continue;
        }
        let footprint = body_fitted_brace_footprint(resolved.solid, &design.cabinet, wall);
        let Some((min_u, max_u, min_v, max_v)) = polygon_bounds(&footprint) else {
            continue;
        };
        const TOLERANCE: f64 = 1.0e-10;
        if min_u <= outer_min_u + TOLERANCE
            && max_u >= outer_max_u - TOLERANCE
            && min_v <= outer_min_v + TOLERANCE
            && max_v >= outer_max_v - TOLERANCE
        {
            return Ok(true);
        }
    }
    Ok(false)
}

/// Split a rectangular panel around a brace that spans from one opposite
/// panel boundary to the other. Such a brace is not a conventional interior
/// hole: the panel surface must be cut into two regions and the brace's
/// underside/side faces become the new fluid boundary.
fn body_fitted_boundary_spanning_brace_regions(
    outer: &[(f64, f64)],
    footprint: &[(f64, f64)],
) -> Option<Vec<Vec<(f64, f64)>>> {
    let (outer_min_u, outer_max_u, outer_min_v, outer_max_v) = polygon_bounds(outer)?;
    let (brace_min_u, brace_max_u, brace_min_v, brace_max_v) = polygon_bounds(footprint)?;
    let mut outer_boundary = outer.to_vec();
    outer_boundary.extend_from_slice(footprint);
    let outer = outer_boundary.as_slice();
    let tolerance = 1.0e-10;
    let covers_u = brace_min_u <= outer_min_u + tolerance && brace_max_u >= outer_max_u - tolerance;
    let covers_v = brace_min_v <= outer_min_v + tolerance && brace_max_v >= outer_max_v - tolerance;

    // A full-width/depth brace is a solid panel strip, not a hole. Remove
    // the covered panel region while retaining any fluid region on the
    // uncovered side(s); its exposed face is supplied by the brace PLC.
    if covers_u && covers_v {
        return Some(Vec::new());
    }
    if covers_u {
        if brace_min_v > outer_min_v + tolerance && brace_max_v >= outer_max_v - tolerance {
            return Some(vec![rectangle_loop_with_outer_boundary(
                outer,
                outer_min_u,
                outer_max_u,
                outer_min_v,
                brace_min_v,
            )]);
        }
        if brace_min_v <= outer_min_v + tolerance && brace_max_v < outer_max_v - tolerance {
            return Some(vec![rectangle_loop_with_outer_boundary(
                outer,
                outer_min_u,
                outer_max_u,
                brace_max_v,
                outer_max_v,
            )]);
        }
    }
    if covers_v {
        if brace_min_u > outer_min_u + tolerance && brace_max_u >= outer_max_u - tolerance {
            return Some(vec![rectangle_loop_with_outer_boundary(
                outer,
                outer_min_u,
                brace_min_u,
                outer_min_v,
                outer_max_v,
            )]);
        }
        if brace_min_u <= outer_min_u + tolerance && brace_max_u < outer_max_u - tolerance {
            return Some(vec![rectangle_loop_with_outer_boundary(
                outer,
                brace_max_u,
                outer_max_u,
                outer_min_v,
                outer_max_v,
            )]);
        }
    }
    let spans_v = brace_min_v <= outer_min_v + tolerance
        && brace_max_v >= outer_max_v - tolerance
        && brace_min_u > outer_min_u + tolerance
        && brace_max_u < outer_max_u - tolerance;
    let spans_u = brace_min_u <= outer_min_u + tolerance
        && brace_max_u >= outer_max_u - tolerance
        && brace_min_v > outer_min_v + tolerance
        && brace_max_v < outer_max_v - tolerance;

    if spans_v {
        Some(vec![
            rectangle_loop_with_outer_boundary(
                outer,
                outer_min_u,
                brace_min_u,
                outer_min_v,
                outer_max_v,
            ),
            rectangle_loop_with_outer_boundary(
                outer,
                brace_max_u,
                outer_max_u,
                outer_min_v,
                outer_max_v,
            ),
        ])
    } else if spans_u {
        Some(vec![
            rectangle_loop_with_outer_boundary(
                outer,
                outer_min_u,
                outer_max_u,
                outer_min_v,
                brace_min_v,
            ),
            rectangle_loop_with_outer_boundary(
                outer,
                outer_min_u,
                outer_max_u,
                brace_max_v,
                outer_max_v,
            ),
        ])
    } else {
        let touches_bottom = brace_min_v <= outer_min_v + tolerance;
        let touches_top = brace_max_v >= outer_max_v - tolerance;
        let touches_left = brace_min_u <= outer_min_u + tolerance;
        let touches_right = brace_max_u >= outer_max_u - tolerance;
        let touched_edges = [touches_bottom, touches_top, touches_left, touches_right]
            .into_iter()
            .filter(|touched| *touched)
            .count();
        if touched_edges != 1 {
            return None;
        }

        // A brace that reaches one panel edge is a notch, not a conventional
        // closed hole.  Keep each returned region convex so the constrained
        // triangulator can assign circular driver/port holes independently.
        if touches_bottom
            && brace_min_u > outer_min_u + tolerance
            && brace_max_u < outer_max_u - tolerance
        {
            Some(vec![
                rectangle_loop_with_outer_boundary(
                    outer,
                    outer_min_u,
                    outer_max_u,
                    brace_max_v,
                    outer_max_v,
                ),
                rectangle_loop_with_outer_boundary(
                    outer,
                    outer_min_u,
                    brace_min_u,
                    outer_min_v,
                    brace_max_v,
                ),
                rectangle_loop_with_outer_boundary(
                    outer,
                    brace_max_u,
                    outer_max_u,
                    outer_min_v,
                    brace_max_v,
                ),
            ])
        } else if touches_top
            && brace_min_u > outer_min_u + tolerance
            && brace_max_u < outer_max_u - tolerance
        {
            Some(vec![
                rectangle_loop_with_outer_boundary(
                    outer,
                    outer_min_u,
                    outer_max_u,
                    outer_min_v,
                    brace_min_v,
                ),
                rectangle_loop_with_outer_boundary(
                    outer,
                    outer_min_u,
                    brace_min_u,
                    brace_min_v,
                    outer_max_v,
                ),
                rectangle_loop_with_outer_boundary(
                    outer,
                    brace_max_u,
                    outer_max_u,
                    brace_min_v,
                    outer_max_v,
                ),
            ])
        } else if touches_left
            && brace_min_v > outer_min_v + tolerance
            && brace_max_v < outer_max_v - tolerance
        {
            Some(vec![
                rectangle_loop_with_outer_boundary(
                    outer,
                    brace_max_u,
                    outer_max_u,
                    outer_min_v,
                    outer_max_v,
                ),
                rectangle_loop_with_outer_boundary(
                    outer,
                    outer_min_u,
                    brace_max_u,
                    outer_min_v,
                    brace_min_v,
                ),
                rectangle_loop_with_outer_boundary(
                    outer,
                    outer_min_u,
                    brace_max_u,
                    brace_max_v,
                    outer_max_v,
                ),
            ])
        } else if touches_right
            && brace_min_v > outer_min_v + tolerance
            && brace_max_v < outer_max_v - tolerance
        {
            Some(vec![
                rectangle_loop_with_outer_boundary(
                    outer,
                    outer_min_u,
                    brace_min_u,
                    outer_min_v,
                    outer_max_v,
                ),
                rectangle_loop_with_outer_boundary(
                    outer,
                    brace_min_u,
                    outer_max_u,
                    outer_min_v,
                    brace_min_v,
                ),
                rectangle_loop_with_outer_boundary(
                    outer,
                    brace_min_u,
                    outer_max_u,
                    brace_max_v,
                    outer_max_v,
                ),
            ])
        } else {
            None
        }
    }
}

/// One triangulated panel region: `(boundary_loop, triangles)`.
type TriangulatedPanelRegion = (Vec<(f64, f64)>, Vec<[usize; 3]>);

fn triangulate_body_fitted_panel_regions(
    outer: &[(f64, f64)],
    brace_footprints: &[Vec<(f64, f64)>],
    holes: &[Vec<(f64, f64)>],
) -> Result<Vec<TriangulatedPanelRegion>, ()> {
    if !brace_footprints.is_empty()
        && let Some(regions) = body_fitted_brace_grid_regions(outer, brace_footprints)
    {
        let mut assigned = vec![false; holes.len()];
        let mut triangulated = Vec::with_capacity(regions.len());
        let mut grid_failed = false;
        for region in regions {
            let hole_refs = holes
                .iter()
                .enumerate()
                .filter_map(|(index, hole)| {
                    if hole_fits_panel_region(hole, &region) {
                        assigned[index] = true;
                        Some(hole.as_slice())
                    } else {
                        None
                    }
                })
                .collect::<Vec<_>>();
            let Ok((vertices, triangles)) =
                triangulate_convex_polygon_with_holes(&region, &hole_refs)
            else {
                grid_failed = true;
                break;
            };
            triangulated.push((vertices, triangles));
        }
        if !grid_failed && !assigned.iter().any(|assigned| !assigned) {
            return Ok(triangulated);
        }
    }

    if brace_footprints.len() > 1 {
        let mut regions = vec![outer.to_vec()];
        for footprint in brace_footprints {
            let mut next_regions = Vec::new();
            let mut split = false;
            for region in regions {
                if let Some(subregions) =
                    body_fitted_boundary_spanning_brace_regions(&region, footprint)
                {
                    next_regions.extend(subregions);
                    split = true;
                } else {
                    next_regions.push(region);
                }
            }
            if !split {
                return Err(());
            }
            regions = next_regions;
        }

        let mut assigned = vec![false; holes.len()];
        let mut triangulated = Vec::with_capacity(regions.len());
        for region in regions {
            let hole_refs = holes
                .iter()
                .enumerate()
                .filter_map(|(index, hole)| {
                    if hole_fits_panel_region(hole, &region) {
                        assigned[index] = true;
                        Some(hole.as_slice())
                    } else {
                        None
                    }
                })
                .collect::<Vec<_>>();
            let (vertices, triangles) =
                triangulate_convex_polygon_with_holes(&region, &hole_refs).map_err(|_| ())?;
            triangulated.push((vertices, triangles));
        }
        if assigned.iter().any(|assigned| !assigned) {
            return Err(());
        }
        return Ok(triangulated);
    }

    let split_regions = (brace_footprints.len() == 1)
        .then(|| body_fitted_boundary_spanning_brace_regions(outer, &brace_footprints[0]));
    if let Some(Some(regions)) = split_regions {
        let mut assigned = vec![false; holes.len()];
        let mut triangulated = Vec::with_capacity(regions.len());
        for region in regions {
            let hole_refs = holes
                .iter()
                .enumerate()
                .filter_map(|(index, hole)| {
                    if hole_fits_panel_region(hole, &region) {
                        assigned[index] = true;
                        Some(hole.as_slice())
                    } else {
                        None
                    }
                })
                .collect::<Vec<_>>();
            let (vertices, triangles) =
                triangulate_convex_polygon_with_holes(&region, &hole_refs).map_err(|_| ())?;
            triangulated.push((vertices, triangles));
        }
        if assigned.iter().any(|assigned| !assigned) {
            return Err(());
        }
        return Ok(triangulated);
    }

    let mut all_holes = holes.to_vec();
    all_holes.extend(brace_footprints.iter().cloned());
    let hole_refs = all_holes.iter().map(Vec::as_slice).collect::<Vec<_>>();
    let (vertices, triangles) =
        triangulate_convex_polygon_with_holes(outer, &hole_refs).map_err(|_| ())?;
    Ok(vec![(vertices, triangles)])
}

fn body_fitted_brace_grid_regions(
    outer: &[(f64, f64)],
    brace_footprints: &[Vec<(f64, f64)>],
) -> Option<Vec<Vec<(f64, f64)>>> {
    const TOLERANCE: f64 = 1.0e-10;
    let (outer_min_u, outer_max_u, outer_min_v, outer_max_v) = polygon_bounds(outer)?;
    let mut us = vec![outer_min_u, outer_max_u];
    let mut vs = vec![outer_min_v, outer_max_v];
    let mut bounds = Vec::with_capacity(brace_footprints.len());
    for footprint in brace_footprints {
        let footprint_bounds = polygon_bounds(footprint)?;
        if footprint_bounds.0 < outer_min_u - TOLERANCE
            || footprint_bounds.1 > outer_max_u + TOLERANCE
            || footprint_bounds.2 < outer_min_v - TOLERANCE
            || footprint_bounds.3 > outer_max_v + TOLERANCE
        {
            return None;
        }
        us.extend([footprint_bounds.0, footprint_bounds.1]);
        vs.extend([footprint_bounds.2, footprint_bounds.3]);
        bounds.push(footprint_bounds);
    }
    us.sort_by(f64::total_cmp);
    us.dedup_by(|left, right| (*left - *right).abs() <= TOLERANCE);
    vs.sort_by(f64::total_cmp);
    vs.dedup_by(|left, right| (*left - *right).abs() <= TOLERANCE);

    let blocked = (0..us.len() - 1)
        .map(|u_index| {
            (0..vs.len() - 1)
                .map(|v_index| {
                    let center = (
                        (us[u_index] + us[u_index + 1]) * 0.5,
                        (vs[v_index] + vs[v_index + 1]) * 0.5,
                    );
                    bounds.iter().any(|&(min_u, max_u, min_v, max_v)| {
                        center.0 > min_u + TOLERANCE
                            && center.0 < max_u - TOLERANCE
                            && center.1 > min_v + TOLERANCE
                            && center.1 < max_v - TOLERANCE
                    })
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let mut visited = vec![vec![false; vs.len() - 1]; us.len() - 1];
    let reverse = polygon_signed_area(outer) < 0.0;
    let mut regions = Vec::new();
    for u_index in 0..us.len() - 1 {
        for v_index in 0..vs.len() - 1 {
            if blocked[u_index][v_index] || visited[u_index][v_index] {
                continue;
            }
            let mut width = 1;
            while u_index + width < us.len() - 1
                && !blocked[u_index + width][v_index]
                && !visited[u_index + width][v_index]
            {
                width += 1;
            }
            let mut height = 1;
            while v_index + height < vs.len() - 1
                && (v_index..v_index + height + 1).all(|current_v| {
                    (u_index..u_index + width).all(|current_u| {
                        !blocked[current_u][current_v] && !visited[current_u][current_v]
                    })
                })
            {
                height += 1;
            }
            for row in &mut visited[u_index..u_index + width] {
                for slot in &mut row[v_index..v_index + height] {
                    *slot = true;
                }
            }
            let mut region = vec![
                (us[u_index], vs[v_index]),
                (us[u_index + width], vs[v_index]),
                (us[u_index + width], vs[v_index + height]),
                (us[u_index], vs[v_index + height]),
            ];
            if reverse {
                region.reverse();
            }
            regions.push(region);
        }
    }
    (!regions.is_empty()).then_some(regions)
}

fn polygon_signed_area(points: &[(f64, f64)]) -> f64 {
    points
        .iter()
        .zip(points.iter().cycle().skip(1))
        .take(points.len())
        .map(|(&(left_u, left_v), &(right_u, right_v))| left_u * right_v - right_u * left_v)
        .sum::<f64>()
        * 0.5
}

fn hole_fits_panel_region(hole: &[(f64, f64)], region: &[(f64, f64)]) -> bool {
    let Some((min_u, max_u, min_v, max_v)) = polygon_bounds(region) else {
        return false;
    };
    const TOLERANCE: f64 = 1.0e-10;
    hole.iter().all(|&(u, v)| {
        u > min_u + TOLERANCE
            && u < max_u - TOLERANCE
            && v > min_v + TOLERANCE
            && v < max_v - TOLERANCE
    })
}

fn polygon_bounds(points: &[(f64, f64)]) -> Option<(f64, f64, f64, f64)> {
    let first = *points.first()?;
    if !first.0.is_finite() || !first.1.is_finite() {
        return None;
    }
    let mut min_u = first.0;
    let mut max_u = first.0;
    let mut min_v = first.1;
    let mut max_v = first.1;
    for &(u, v) in &points[1..] {
        if !u.is_finite() || !v.is_finite() {
            return None;
        }
        min_u = min_u.min(u);
        max_u = max_u.max(u);
        min_v = min_v.min(v);
        max_v = max_v.max(v);
    }
    Some((min_u, max_u, min_v, max_v))
}

fn rectangle_loop_with_outer_boundary(
    outer: &[(f64, f64)],
    min_u: f64,
    max_u: f64,
    min_v: f64,
    max_v: f64,
) -> Vec<(f64, f64)> {
    const TOLERANCE: f64 = 1.0e-10;
    let on_horizontal = |point: (f64, f64), value: f64| (point.1 - value).abs() <= TOLERANCE;
    let on_vertical = |point: (f64, f64), value: f64| (point.0 - value).abs() <= TOLERANCE;
    let edge = |horizontal: bool, value: f64, start: f64, end: f64, ascending: bool| {
        let mut points = outer
            .iter()
            .copied()
            .filter(|point| {
                (if horizontal {
                    on_horizontal(*point, value)
                } else {
                    on_vertical(*point, value)
                }) && {
                    let coordinate = if horizontal { point.0 } else { point.1 };
                    coordinate >= start - TOLERANCE && coordinate <= end + TOLERANCE
                }
            })
            .collect::<Vec<_>>();
        points.push(if horizontal {
            (start, value)
        } else {
            (value, start)
        });
        points.push(if horizontal {
            (end, value)
        } else {
            (value, end)
        });
        points.sort_by(|a, b| {
            let a_coordinate = if horizontal { a.0 } else { a.1 };
            let b_coordinate = if horizontal { b.0 } else { b.1 };
            a_coordinate
                .partial_cmp(&b_coordinate)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        points.dedup_by(|a, b| (a.0 - b.0).abs() <= TOLERANCE && (a.1 - b.1).abs() <= TOLERANCE);
        if !ascending {
            points.reverse();
        }
        points
    };

    let edges = [
        edge(true, min_v, min_u, max_u, true),
        edge(false, max_u, min_v, max_v, true),
        edge(true, max_v, min_u, max_u, false),
        edge(false, min_u, min_v, max_v, false),
    ];
    let mut loop_points = Vec::new();
    for edge in edges {
        for point in edge {
            if loop_points.last().is_none_or(|last: &(f64, f64)| {
                (last.0 - point.0).abs() > TOLERANCE || (last.1 - point.1).abs() > TOLERANCE
            }) {
                loop_points.push(point);
            }
        }
    }
    if loop_points.len() > 1
        && (loop_points[0].0 - loop_points.last().unwrap().0).abs() <= TOLERANCE
        && (loop_points[0].1 - loop_points.last().unwrap().1).abs() <= TOLERANCE
    {
        loop_points.pop();
    }
    loop_points
}

fn body_fitted_brace_contact_vertices(
    solid: AxisAlignedSolid,
    wall: InteriorWall,
) -> [[f64; 3]; 4] {
    let [xmin, ymin, zmin] = solid.min_m;
    let [xmax, ymax, zmax] = solid.max_m;
    match wall {
        InteriorWall::Top => [
            [xmin, ymax, zmin],
            [xmax, ymax, zmin],
            [xmax, ymax, zmax],
            [xmin, ymax, zmax],
        ],
        InteriorWall::Bottom => [
            [xmin, ymin, zmin],
            [xmin, ymin, zmax],
            [xmax, ymin, zmax],
            [xmax, ymin, zmin],
        ],
        InteriorWall::Left => [
            [xmin, ymin, zmin],
            [xmin, ymax, zmin],
            [xmin, ymax, zmax],
            [xmin, ymin, zmax],
        ],
        InteriorWall::Right => [
            [xmax, ymin, zmin],
            [xmax, ymin, zmax],
            [xmax, ymax, zmax],
            [xmax, ymax, zmin],
        ],
        InteriorWall::Back => [
            [xmin, ymin, zmin],
            [xmax, ymin, zmin],
            [xmax, ymax, zmin],
            [xmin, ymax, zmin],
        ],
        InteriorWall::Front => [
            [xmin, ymin, zmax],
            [xmax, ymin, zmax],
            [xmax, ymax, zmax],
            [xmin, ymax, zmax],
        ],
    }
}

fn body_fitted_push_brace_faces(
    nodes: &mut Vec<Point>,
    facets: &mut Vec<([usize; 3], i32)>,
    solid: AxisAlignedSolid,
    cabinet: &CabinetDesign,
    attached_wall: InteriorWall,
    marker: i32,
) {
    let [xmin, ymin, zmin] = solid.min_m;
    let [xmax, ymax, zmax] = solid.max_m;
    let corners = [
        [xmin, ymin, zmin],
        [xmax, ymin, zmin],
        [xmax, ymax, zmin],
        [xmin, ymax, zmin],
        [xmin, ymin, zmax],
        [xmax, ymin, zmax],
        [xmax, ymax, zmax],
        [xmin, ymax, zmax],
    ]
    .map(|point| body_fitted_insert_node(nodes, point));
    let faces = [
        (InteriorWall::Back, [0, 1, 2, 3]),
        (InteriorWall::Front, [4, 7, 6, 5]),
        (InteriorWall::Bottom, [0, 4, 5, 1]),
        (InteriorWall::Right, [1, 5, 6, 2]),
        (InteriorWall::Top, [2, 6, 7, 3]),
        (InteriorWall::Left, [3, 7, 4, 0]),
    ];
    for (wall, [a, b, c, d]) in faces {
        if wall == attached_wall || body_fitted_brace_touches_wall(solid, cabinet, wall) {
            continue;
        }
        facets.push(([corners[a], corners[b], corners[c]], marker));
        facets.push(([corners[a], corners[c], corners[d]], marker));
    }
}

fn body_fitted_refine_surface_facets(
    nodes: &[Point],
    facets: &[([usize; 3], i32)],
) -> Vec<([usize; 3], i32)> {
    let mut pending = facets.to_vec();
    let mut refined = Vec::with_capacity(facets.len());
    while let Some((facet, marker)) = pending.pop() {
        let split = [
            (facet[0], facet[1], facet[2]),
            (facet[1], facet[2], facet[0]),
            (facet[2], facet[0], facet[1]),
        ]
        .into_iter()
        .find_map(|(start, end, opposite)| {
            nodes
                .iter()
                .enumerate()
                .find(|(index, point)| {
                    *index != start
                        && *index != end
                        && *index != opposite
                        && body_fitted_point_on_segment(**point, nodes[start], nodes[end])
                })
                .map(|(index, _)| (start, end, opposite, index))
        });
        if let Some((start, end, opposite, split)) = split {
            pending.push(([start, split, opposite], marker));
            pending.push(([split, end, opposite], marker));
        } else {
            refined.push((facet, marker));
        }
    }
    refined
}

fn body_fitted_point_on_segment(point: Point, start: Point, end: Point) -> bool {
    let direction = [end.x - start.x, end.y - start.y, end.z - start.z];
    let offset = [point.x - start.x, point.y - start.y, point.z - start.z];
    let length_squared = direction.iter().map(|value| value * value).sum::<f64>();
    if length_squared <= 1.0e-24 {
        return false;
    }
    let t = direction
        .iter()
        .zip(offset)
        .map(|(direction, offset)| direction * offset)
        .sum::<f64>()
        / length_squared;
    if t <= 1.0e-10 || t >= 1.0 - 1.0e-10 {
        return false;
    }
    direction
        .iter()
        .zip(offset)
        .all(|(direction, offset)| (offset - t * direction).abs() <= 1.0e-10)
}

fn body_fitted_orient_surface_facets(nodes: &[Point], facets: &mut [([usize; 3], i32)]) {
    let mut edge_users = BTreeMap::<[usize; 2], Vec<(usize, i32)>>::new();
    for (facet_index, (facet, _)) in facets.iter().enumerate() {
        for [left, right] in [
            [facet[0], facet[1]],
            [facet[1], facet[2]],
            [facet[2], facet[0]],
        ] {
            let edge = if left < right {
                [left, right]
            } else {
                [right, left]
            };
            let direction = if left < right { 1 } else { -1 };
            edge_users
                .entry(edge)
                .or_default()
                .push((facet_index, direction));
        }
    }

    let mut flips = vec![None; facets.len()];
    for seed in 0..facets.len() {
        if flips[seed].is_some() {
            continue;
        }
        flips[seed] = Some(false);
        let mut pending = vec![seed];
        while let Some(facet_index) = pending.pop() {
            let facet = facets[facet_index].0;
            for [left, right] in [
                [facet[0], facet[1]],
                [facet[1], facet[2]],
                [facet[2], facet[0]],
            ] {
                let edge = if left < right {
                    [left, right]
                } else {
                    [right, left]
                };
                let current_original_direction = edge_users[&edge]
                    .iter()
                    .find_map(|(index, direction)| (*index == facet_index).then_some(*direction))
                    .expect("surface edge must reference its facet");
                let current_direction = if flips[facet_index].unwrap_or(false) {
                    -current_original_direction
                } else {
                    current_original_direction
                };
                for (neighbor, neighbor_original_direction) in &edge_users[&edge] {
                    if *neighbor == facet_index {
                        continue;
                    }
                    let should_flip = *neighbor_original_direction == current_direction;
                    if let Some(existing) = flips[*neighbor] {
                        debug_assert_eq!(existing, should_flip);
                    } else {
                        flips[*neighbor] = Some(should_flip);
                        pending.push(*neighbor);
                    }
                }
            }
        }
    }

    for (facet, flip) in facets.iter_mut().zip(flips) {
        if flip.unwrap_or(false) {
            facet.0.swap(1, 2);
        }
    }

    let signed_volume = facets
        .iter()
        .map(|([a, b, c], _)| {
            let a = nodes[*a];
            let b = nodes[*b];
            let c = nodes[*c];
            a.x * (b.y * c.z - b.z * c.y)
                + a.y * (b.z * c.x - b.x * c.z)
                + a.z * (b.x * c.y - b.y * c.x)
        })
        .sum::<f64>();
    if signed_volume < 0.0 {
        for (facet, _) in facets {
            facet.swap(1, 2);
        }
    }
}

fn body_fitted_ring_segments(
    radius_m: f64,
    config: ProductInteriorMeshConfig,
) -> Result<usize, ProductInteriorMeshError> {
    let requested = (std::f64::consts::TAU * radius_m / config.max_element_size_m)
        .ceil()
        .max(MIN_BODY_FITTED_RING_SEGMENTS as f64) as usize;
    let segments = requested.div_ceil(4) * 4;
    let maximum = config.max_cells_per_axis.saturating_mul(4);
    if segments > maximum {
        return Err(ProductInteriorMeshError::BodyFittedResolutionLimit {
            requested: segments,
            maximum,
        });
    }
    Ok(segments)
}

fn same_triangle_nodes(nodes: &[usize], expected: &[usize; 3]) -> bool {
    if nodes.len() != 3 {
        return false;
    }
    let mut actual = [nodes[0], nodes[1], nodes[2]];
    let mut expected = *expected;
    actual.sort_unstable();
    expected.sort_unstable();
    actual == expected
}

fn body_fitted_panel_dimensions(cabinet: &CabinetDesign, wall: InteriorWall) -> (f64, f64) {
    match wall {
        InteriorWall::Front | InteriorWall::Back => (cabinet.box_width_m, cabinet.box_height_m),
        InteriorWall::Left | InteriorWall::Right => (cabinet.box_depth_m, cabinet.box_height_m),
        InteriorWall::Bottom | InteriorWall::Top => (cabinet.box_width_m, cabinet.box_depth_m),
    }
}

fn body_fitted_panel_boundary(
    cabinet: &CabinetDesign,
    wall: InteriorWall,
    ring_segments: usize,
) -> (Vec<[f64; 2]>, Vec<InteriorWall>) {
    debug_assert_eq!(ring_segments % 4, 0);
    let per_side = ring_segments / 4;
    let (width, height, side_walls) = match wall {
        InteriorWall::Front => (
            cabinet.box_width_m,
            cabinet.box_height_m,
            [
                InteriorWall::Bottom,
                InteriorWall::Right,
                InteriorWall::Top,
                InteriorWall::Left,
            ],
        ),
        InteriorWall::Back => (
            cabinet.box_width_m,
            cabinet.box_height_m,
            [
                InteriorWall::Bottom,
                InteriorWall::Left,
                InteriorWall::Top,
                InteriorWall::Right,
            ],
        ),
        InteriorWall::Right => (
            cabinet.box_depth_m,
            cabinet.box_height_m,
            [
                InteriorWall::Bottom,
                InteriorWall::Back,
                InteriorWall::Top,
                InteriorWall::Front,
            ],
        ),
        InteriorWall::Left => (
            cabinet.box_depth_m,
            cabinet.box_height_m,
            [
                InteriorWall::Bottom,
                InteriorWall::Front,
                InteriorWall::Top,
                InteriorWall::Back,
            ],
        ),
        InteriorWall::Top => (
            cabinet.box_width_m,
            cabinet.box_depth_m,
            [
                InteriorWall::Front,
                InteriorWall::Right,
                InteriorWall::Back,
                InteriorWall::Left,
            ],
        ),
        InteriorWall::Bottom => (
            cabinet.box_width_m,
            cabinet.box_depth_m,
            [
                InteriorWall::Back,
                InteriorWall::Right,
                InteriorWall::Front,
                InteriorWall::Left,
            ],
        ),
    };
    let half_width = width * 0.5;
    let half_height = height * 0.5;
    let mut points = Vec::with_capacity(ring_segments);
    let mut markers = Vec::with_capacity(ring_segments);
    for step in 0..per_side {
        let t = step as f64 / per_side as f64;
        points.push([-half_width + width * t, -half_height]);
        markers.push(side_walls[0]);
    }
    for step in 0..per_side {
        let t = step as f64 / per_side as f64;
        points.push([half_width, -half_height + height * t]);
        markers.push(side_walls[1]);
    }
    for step in 0..per_side {
        let t = step as f64 / per_side as f64;
        points.push([half_width - width * t, half_height]);
        markers.push(side_walls[2]);
    }
    for step in 0..per_side {
        let t = step as f64 / per_side as f64;
        points.push([-half_width, half_height - height * t]);
        markers.push(side_walls[3]);
    }
    (points, markers)
}

fn body_fitted_panel_coordinates(
    cabinet: &CabinetDesign,
    wall: InteriorWall,
    [x, y, z]: [f64; 3],
) -> [f64; 2] {
    match wall {
        InteriorWall::Front => [x, y],
        InteriorWall::Back => [-x, y],
        InteriorWall::Right => [-z - cabinet.box_depth_m * 0.5, y],
        InteriorWall::Left => [z + cabinet.box_depth_m * 0.5, y],
        InteriorWall::Top => [x, -z - cabinet.box_depth_m * 0.5],
        InteriorWall::Bottom => [x, z + cabinet.box_depth_m * 0.5],
    }
}

fn body_fitted_panel_point(
    cabinet: &CabinetDesign,
    wall: InteriorWall,
    [u, v]: [f64; 2],
) -> [f64; 3] {
    match wall {
        InteriorWall::Front => [u, v, 0.0],
        InteriorWall::Back => [-u, v, -cabinet.box_depth_m],
        InteriorWall::Right => [cabinet.box_width_m * 0.5, v, -u - cabinet.box_depth_m * 0.5],
        InteriorWall::Left => [-cabinet.box_width_m * 0.5, v, u - cabinet.box_depth_m * 0.5],
        InteriorWall::Top => [
            u,
            cabinet.box_height_m * 0.5,
            -v - cabinet.box_depth_m * 0.5,
        ],
        InteriorWall::Bottom => [
            u,
            -cabinet.box_height_m * 0.5,
            v - cabinet.box_depth_m * 0.5,
        ],
    }
}

fn body_fitted_inward_offset(cabinet: &CabinetDesign, wall: InteriorWall) -> [f64; 3] {
    match wall {
        InteriorWall::Front => [0.0, 0.0, -cabinet.box_depth_m],
        InteriorWall::Back => [0.0, 0.0, cabinet.box_depth_m],
        InteriorWall::Right => [-cabinet.box_width_m, 0.0, 0.0],
        InteriorWall::Left => [cabinet.box_width_m, 0.0, 0.0],
        InteriorWall::Top => [0.0, -cabinet.box_height_m, 0.0],
        InteriorWall::Bottom => [0.0, cabinet.box_height_m, 0.0],
    }
}

fn body_fitted_product_mesh_result(
    design: &ProductDesign,
    config: ProductInteriorMeshConfig,
    ring_segments: usize,
    mesh: FemMesh,
) -> Result<ProductInteriorMesh, ProductInteriorMeshError> {
    let kinds = InteriorWall::ALL
        .into_iter()
        .map(ProductInteriorPatchKind::RigidWall)
        .chain(
            design
                .drivers
                .iter()
                .enumerate()
                .map(|(driver_index, _)| ProductInteriorPatchKind::DriverRear { driver_index }),
        )
        .chain(
            design
                .waveguide
                .as_ref()
                .map(|_| ProductInteriorPatchKind::WaveguideWall { driver_index: 0 }),
        )
        .chain(
            design
                .cabinet
                .port
                .is_some()
                .then_some(ProductInteriorPatchKind::PortMouth),
        )
        .chain(
            design
                .cabinet
                .port
                .is_some()
                .then_some(ProductInteriorPatchKind::PortWall),
        )
        .chain(
            design
                .cabinet
                .braces
                .iter()
                .enumerate()
                .map(|(brace_index, _)| ProductInteriorPatchKind::BraceWall { brace_index }),
        )
        .collect::<Vec<_>>();
    let mut patches = Vec::with_capacity(kinds.len());
    let mut patch_area_m2 = BTreeMap::new();
    for kind in kinds {
        let marker = marker_for(kind)?;
        let boundary_faces = mesh
            .boundaries
            .iter()
            .enumerate()
            .filter_map(|(index, face)| (face.marker == marker).then_some(index))
            .collect::<Vec<_>>();
        if boundary_faces.is_empty() {
            if matches!(kind, ProductInteriorPatchKind::WaveguideWall { .. }) {
                return Err(ProductInteriorMeshError::MissingPatch { kind });
            }
            if let ProductInteriorPatchKind::RigidWall(wall) = kind
                && body_fitted_wall_is_covered_by_brace(design, wall)?
            {
                continue;
            }
            return Err(ProductInteriorMeshError::MissingPatch { kind });
        }
        let area_m2 = boundary_faces
            .iter()
            .map(|&face| boundary_area(&mesh, face))
            .sum::<Result<f64, _>>()?;
        patch_area_m2.insert(patch_name(kind), area_m2);
        patches.push(ProductInteriorPatch {
            kind,
            marker,
            boundary_faces,
            area_m2,
            canonical_normal_m: canonical_patch_normal(design, kind),
        });
    }
    let diagnostics = ProductInteriorMeshDiagnostics {
        node_count: mesh.num_nodes(),
        tetrahedron_count: mesh.num_elements(),
        boundary_face_count: mesh.boundaries.len(),
        fluid_volume_m3: tetrahedral_volume(&mesh),
        expected_box_volume_m3: design.cabinet.volume_m3(),
        patch_area_m2,
    };
    Ok(ProductInteriorMesh {
        mesh,
        patches,
        diagnostics,
        geometry_fingerprint: format!(
            "{}-body-fitted-driver-{}",
            geometry_fingerprint(design, config),
            ring_segments
        ),
    })
}

/// Build a deterministic rectangular sealed-cavity mesh with one distinct
/// rear patch per front-facing driver.
///
/// The front plane is `z = 0`; the cavity extends to `z = -box_depth_m`.
/// Driver patches are grid-aligned approximations selected by face centroid.
/// This is sufficient to establish stable per-driver FEM ownership and mesh
/// evidence, but it is not yet the final circular-rim body-fitted aperture
/// implementation.
pub fn build_sealed_product_interior_mesh(
    design: &ProductDesign,
    config: ProductInteriorMeshConfig,
) -> Result<ProductInteriorMesh, ProductInteriorMeshError> {
    design.validate()?;
    config.validate()?;
    if let Some(port) = design.cabinet.port {
        // Resolve before returning the capability error so callers receive
        // the more useful missing-solid-geometry diagnostic for legacy input.
        let _port_geometry = ResolvedPortGeometry::from_design(port)?;
        return Err(ProductInteriorMeshError::ResolvedPortUnsupported);
    }
    let cabinet = &design.cabinet;
    let resolved_braces: Vec<_> = cabinet
        .braces
        .iter()
        .copied()
        .enumerate()
        .map(|(index, brace)| ResolvedBraceGeometry::from_design(cabinet, brace, index))
        .collect::<Result<_, _>>()?;
    for (index, brace) in resolved_braces.iter().enumerate() {
        let cross_section = cabinet.braces[index]
            .cross_section
            .expect("resolved braces require a cross-section");
        if cross_section.width_m < config.max_element_size_m
            || cross_section.depth_m < config.max_element_size_m
        {
            return Err(ProductInteriorMeshError::BraceUnresolved { index });
        }
        for (other_index, other) in resolved_braces.iter().take(index).enumerate() {
            if brace.solid.overlaps_volume(other.solid) {
                return Err(ProductInteriorMeshError::BraceSolidIntersection {
                    first: other_index,
                    second: index,
                });
            }
        }
    }
    for (index, driver) in design.drivers.iter().enumerate() {
        if !driver.is_axis_aligned() {
            return Err(ProductInteriorMeshError::BodyFittedDriverLayoutUnsupported);
        }
        let Some(wall) = driver_wall(driver) else {
            return Err(ProductInteriorMeshError::DriverNotOnCabinetWall { index });
        };
        if wall == InteriorWall::Front
            && (driver.position_m[0].abs() + driver.radius_m > cabinet.box_width_m * 0.5
                || driver.position_m[1].abs() + driver.radius_m > cabinet.box_height_m * 0.5)
        {
            return Err(ProductInteriorMeshError::DriverOutsideInterior { index });
        }
    }

    let nx = config.cells_for(cabinet.box_width_m)?;
    let ny = config.cells_for(cabinet.box_height_m)?;
    let nz = config.cells_for(cabinet.box_depth_m)?;
    let mut mesh = box_mesh_tetrahedra(
        -cabinet.box_width_m * 0.5,
        cabinet.box_width_m * 0.5,
        -cabinet.box_height_m * 0.5,
        cabinet.box_height_m * 0.5,
        -cabinet.box_depth_m,
        0.0,
        nx,
        ny,
        nz,
    );
    if !resolved_braces.is_empty() {
        mesh = carve_brace_solids(&mesh, &resolved_braces)?;
    }

    let mut patch_faces: BTreeMap<ProductInteriorPatchKind, Vec<usize>> = BTreeMap::new();
    for wall in InteriorWall::ALL {
        patch_faces.insert(ProductInteriorPatchKind::RigidWall(wall), Vec::new());
    }
    for index in 0..design.drivers.len() {
        patch_faces.insert(
            ProductInteriorPatchKind::DriverRear {
                driver_index: index,
            },
            Vec::new(),
        );
    }
    for index in 0..resolved_braces.len() {
        patch_faces.insert(
            ProductInteriorPatchKind::BraceWall { brace_index: index },
            Vec::new(),
        );
    }

    for face_index in 0..mesh.boundaries.len() {
        let (x, y, z) = boundary_centroid(&mesh, face_index)?;
        let kind = if let Some(wall) = outer_wall_at(cabinet, [x, y, z]) {
            driver_at(design, wall, [x, y, z]).unwrap_or(ProductInteriorPatchKind::RigidWall(wall))
        } else if let Some(brace_index) =
            brace_at(&resolved_braces, [x, y, z], config.max_element_size_m)
        {
            ProductInteriorPatchKind::BraceWall { brace_index }
        } else {
            return Err(ProductInteriorMeshError::InvalidBoundaryFace { face: face_index });
        };
        let face = mesh
            .boundaries
            .get_mut(face_index)
            .expect("face index was taken from this mesh");
        face.boundary_type = BoundaryType::Neumann;
        face.marker = marker_for(kind)?;
        patch_faces
            .get_mut(&kind)
            .expect("all typed patches are initialized")
            .push(face_index);
    }

    let mut patches = Vec::with_capacity(patch_faces.len());
    let mut patch_area_m2 = BTreeMap::new();
    for (kind, faces) in patch_faces {
        if faces.is_empty() {
            return match kind {
                ProductInteriorPatchKind::DriverRear { driver_index } => {
                    Err(ProductInteriorMeshError::DriverPatchEmpty {
                        index: driver_index,
                    })
                }
                ProductInteriorPatchKind::BraceWall { brace_index } => {
                    Err(ProductInteriorMeshError::BraceUnresolved { index: brace_index })
                }
                _ => Err(ProductInteriorMeshError::MissingPatch { kind }),
            };
        }
        let area_m2 = faces
            .iter()
            .map(|&face| boundary_area(&mesh, face))
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .sum();
        patch_area_m2.insert(patch_name(kind), area_m2);
        patches.push(ProductInteriorPatch {
            kind,
            marker: marker_for(kind)?,
            boundary_faces: faces,
            area_m2,
            canonical_normal_m: canonical_patch_normal(design, kind),
        });
    }

    let diagnostics = ProductInteriorMeshDiagnostics {
        node_count: mesh.nodes.len(),
        tetrahedron_count: mesh.elements.len(),
        boundary_face_count: mesh.boundaries.len(),
        fluid_volume_m3: tetrahedral_volume(&mesh),
        expected_box_volume_m3: cabinet.volume_m3(),
        patch_area_m2,
    };
    Ok(ProductInteriorMesh {
        mesh,
        patches,
        diagnostics,
        geometry_fingerprint: geometry_fingerprint(design, config),
    })
}

/// Build the ported product topology by removing tube-wall tetrahedra from the
/// deterministic sealed background.  The remaining bore is continuous with
/// the enclosure and opens at `z = 0`.
pub fn build_ported_product_interior_mesh(
    design: &ProductDesign,
    config: ProductInteriorMeshConfig,
) -> Result<ProductInteriorMesh, ProductInteriorMeshError> {
    design.validate()?;
    let port = design
        .cabinet
        .port
        .ok_or(ProductInteriorMeshError::ResolvedPortUnsupported)?;
    let geometry = ResolvedPortGeometry::from_design(port)?;
    let mut sealed_design = design.clone();
    sealed_design.cabinet.port = None;
    let sealed = build_sealed_product_interior_mesh(&sealed_design, config)?;
    let resolved_braces: Vec<_> = design
        .cabinet
        .braces
        .iter()
        .copied()
        .enumerate()
        .map(|(index, brace)| ResolvedBraceGeometry::from_design(&design.cabinet, brace, index))
        .collect::<Result<_, _>>()?;
    for (brace_index, brace) in resolved_braces.iter().enumerate() {
        if brace_intersects_port(brace.solid, geometry) {
            return Err(ProductInteriorMeshError::PortBraceIntersection { brace_index });
        }
    }
    let mut mesh = carve_port_wall(&sealed.mesh, geometry)?;
    let cabinet = &design.cabinet;
    let mut patch_faces = BTreeMap::<ProductInteriorPatchKind, Vec<usize>>::new();
    for wall in InteriorWall::ALL {
        patch_faces.insert(ProductInteriorPatchKind::RigidWall(wall), Vec::new());
    }
    for driver_index in 0..design.drivers.len() {
        patch_faces.insert(
            ProductInteriorPatchKind::DriverRear { driver_index },
            Vec::new(),
        );
    }
    patch_faces.insert(ProductInteriorPatchKind::PortMouth, Vec::new());
    patch_faces.insert(ProductInteriorPatchKind::PortWall, Vec::new());
    for brace_index in 0..resolved_braces.len() {
        patch_faces.insert(
            ProductInteriorPatchKind::BraceWall { brace_index },
            Vec::new(),
        );
    }

    for face_index in 0..mesh.boundaries.len() {
        let (x, y, z) = boundary_centroid(&mesh, face_index)?;
        let radius = ((x - geometry.mouth_center_m[0]).powi(2)
            + (y - geometry.mouth_center_m[1]).powi(2))
        .sqrt();
        let kind = if z.abs() <= FRONT_PLANE_TOLERANCE_M && radius <= geometry.bore_radius_m {
            ProductInteriorPatchKind::PortMouth
        } else if z < -FRONT_PLANE_TOLERANCE_M
            && z >= -geometry.length_m - config.max_element_size_m
            && radius <= geometry.outer_radius_m + config.max_element_size_m
        {
            ProductInteriorPatchKind::PortWall
        } else if let Some(wall) = outer_wall_at(cabinet, [x, y, z]) {
            driver_at(design, wall, [x, y, z]).unwrap_or(ProductInteriorPatchKind::RigidWall(wall))
        } else if let Some(brace_index) =
            brace_at(&resolved_braces, [x, y, z], config.max_element_size_m)
        {
            ProductInteriorPatchKind::BraceWall { brace_index }
        } else {
            return Err(ProductInteriorMeshError::InvalidBoundaryFace { face: face_index });
        };
        mesh.boundaries[face_index].boundary_type = BoundaryType::Neumann;
        mesh.boundaries[face_index].marker = marker_for(kind)?;
        patch_faces
            .get_mut(&kind)
            .expect("patch registry")
            .push(face_index);
    }

    let mut patches = Vec::new();
    let mut patch_area_m2 = BTreeMap::new();
    for (kind, faces) in patch_faces {
        if faces.is_empty() {
            return Err(match kind {
                ProductInteriorPatchKind::DriverRear { driver_index } => {
                    ProductInteriorMeshError::DriverPatchEmpty {
                        index: driver_index,
                    }
                }
                ProductInteriorPatchKind::PortMouth => ProductInteriorMeshError::PortMouthEmpty,
                ProductInteriorPatchKind::PortWall => ProductInteriorMeshError::PortWallUnresolved,
                ProductInteriorPatchKind::BraceWall { brace_index } => {
                    ProductInteriorMeshError::BraceUnresolved { index: brace_index }
                }
                _ => ProductInteriorMeshError::MissingPatch { kind },
            });
        }
        let area_m2 = faces
            .iter()
            .map(|&face| boundary_area(&mesh, face))
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .sum();
        patch_area_m2.insert(patch_name(kind), area_m2);
        patches.push(ProductInteriorPatch {
            kind,
            marker: marker_for(kind)?,
            boundary_faces: faces,
            area_m2,
            canonical_normal_m: canonical_patch_normal(design, kind),
        });
    }
    Ok(ProductInteriorMesh {
        diagnostics: ProductInteriorMeshDiagnostics {
            node_count: mesh.nodes.len(),
            tetrahedron_count: mesh.elements.len(),
            boundary_face_count: mesh.boundaries.len(),
            fluid_volume_m3: tetrahedral_volume(&mesh),
            expected_box_volume_m3: cabinet.volume_m3(),
            patch_area_m2,
        },
        mesh,
        patches,
        geometry_fingerprint: geometry_fingerprint(design, config),
    })
}

fn carve_port_wall(
    mesh: &FemMesh,
    port: ResolvedPortGeometry,
) -> Result<FemMesh, ProductInteriorMeshError> {
    let mut kept = Vec::new();
    for element in &mesh.elements {
        if element.element_type != ElementType::Tetrahedron || element.nodes.len() != 4 {
            return Err(ProductInteriorMeshError::InvalidMeshConfig(
                "background mesh must be P1 tetrahedral",
            ));
        }
        let mut centroid = [0.0; 3];
        for &node in &element.nodes {
            let point = &mesh.nodes[node];
            centroid[0] += point.x * 0.25;
            centroid[1] += point.y * 0.25;
            centroid[2] += point.z * 0.25;
        }
        let radius = ((centroid[0] - port.mouth_center_m[0]).powi(2)
            + (centroid[1] - port.mouth_center_m[1]).powi(2))
        .sqrt();
        if !(centroid[2] < 0.0
            && centroid[2] >= -port.length_m
            && radius >= port.bore_radius_m
            && radius <= port.outer_radius_m)
        {
            kept.push(element.nodes.clone());
        }
    }
    if kept.len() == mesh.elements.len() {
        return Err(ProductInteriorMeshError::PortWallUnresolved);
    }
    let mut used = vec![false; mesh.nodes.len()];
    for nodes in &kept {
        for &node in nodes {
            used[node] = true;
        }
    }
    let mut compact = FemMesh::new(3);
    let mut map = vec![usize::MAX; mesh.nodes.len()];
    for (old, &is_used) in used.iter().enumerate() {
        if is_used {
            map[old] = compact.add_node(mesh.nodes[old]);
        }
    }
    for nodes in kept {
        compact.add_element(
            ElementType::Tetrahedron,
            nodes.into_iter().map(|node| map[node]).collect(),
        );
    }
    compact.detect_boundaries();
    if !is_single_fluid_component(&mut compact) {
        return Err(ProductInteriorMeshError::DisconnectedFluidDomain);
    }
    Ok(compact)
}

fn carve_brace_solids(
    mesh: &FemMesh,
    braces: &[ResolvedBraceGeometry],
) -> Result<FemMesh, ProductInteriorMeshError> {
    let mut kept = Vec::new();
    let mut removed = vec![0usize; braces.len()];
    for element in &mesh.elements {
        if element.element_type != ElementType::Tetrahedron || element.nodes.len() != 4 {
            return Err(ProductInteriorMeshError::InvalidMeshConfig(
                "background mesh must be P1 tetrahedral",
            ));
        }
        let centroid = element_centroid(mesh, &element.nodes)?;
        if let Some(index) = braces
            .iter()
            .position(|brace| brace.solid.contains(centroid))
        {
            removed[index] += 1;
        } else {
            kept.push(element.nodes.clone());
        }
    }
    if let Some((index, _)) = removed.iter().enumerate().find(|(_, count)| **count == 0) {
        return Err(ProductInteriorMeshError::BraceUnresolved { index });
    }
    let mut compact = compact_tetrahedral_mesh(mesh, kept)?;
    if !is_single_fluid_component(&mut compact) {
        return Err(ProductInteriorMeshError::DisconnectedFluidDomain);
    }
    Ok(compact)
}

fn compact_tetrahedral_mesh(
    mesh: &FemMesh,
    elements: Vec<Vec<usize>>,
) -> Result<FemMesh, ProductInteriorMeshError> {
    let mut used = vec![false; mesh.nodes.len()];
    for nodes in &elements {
        for &node in nodes {
            let is_valid_node =
                used.get_mut(node)
                    .ok_or(ProductInteriorMeshError::InvalidMeshConfig(
                        "background mesh references an invalid node",
                    ))?;
            *is_valid_node = true;
        }
    }
    let mut compact = FemMesh::new(3);
    let mut map = vec![usize::MAX; mesh.nodes.len()];
    for (old, &is_used) in used.iter().enumerate() {
        if is_used {
            map[old] = compact.add_node(mesh.nodes[old]);
        }
    }
    for nodes in elements {
        compact.add_element(
            ElementType::Tetrahedron,
            nodes.into_iter().map(|node| map[node]).collect(),
        );
    }
    compact.detect_boundaries();
    Ok(compact)
}

fn element_centroid(mesh: &FemMesh, nodes: &[usize]) -> Result<[f64; 3], ProductInteriorMeshError> {
    if nodes.len() != 4 {
        return Err(ProductInteriorMeshError::InvalidMeshConfig(
            "background mesh must be P1 tetrahedral",
        ));
    }
    let mut centroid = [0.0; 3];
    for &node in nodes {
        let point = mesh
            .nodes
            .get(node)
            .ok_or(ProductInteriorMeshError::InvalidMeshConfig(
                "background mesh references an invalid node",
            ))?;
        centroid[0] += point.x * 0.25;
        centroid[1] += point.y * 0.25;
        centroid[2] += point.z * 0.25;
    }
    Ok(centroid)
}

fn brace_at(braces: &[ResolvedBraceGeometry], point: [f64; 3], margin_m: f64) -> Option<usize> {
    braces
        .iter()
        .position(|brace| brace.solid.contains_expanded(point, margin_m))
}

fn brace_intersects_port(brace: AxisAlignedSolid, port: ResolvedPortGeometry) -> bool {
    let z_min = -port.length_m;
    let z_max = 0.0;
    if brace.max_m[2] <= z_min || brace.min_m[2] >= z_max {
        return false;
    }
    let closest_x = port.mouth_center_m[0].clamp(brace.min_m[0], brace.max_m[0]);
    let closest_y = port.mouth_center_m[1].clamp(brace.min_m[1], brace.max_m[1]);
    let dx = closest_x - port.mouth_center_m[0];
    let dy = closest_y - port.mouth_center_m[1];
    dx.mul_add(dx, dy * dy) < port.outer_radius_m * port.outer_radius_m
}

fn is_single_fluid_component(mesh: &mut FemMesh) -> bool {
    if mesh.elements.is_empty() {
        return false;
    }
    let neighbors = mesh.element_neighbors().to_vec();
    let mut seen = vec![false; neighbors.len()];
    let mut stack = vec![0usize];
    seen[0] = true;
    while let Some(element) = stack.pop() {
        for &neighbor in &neighbors[element] {
            if !seen[neighbor] {
                seen[neighbor] = true;
                stack.push(neighbor);
            }
        }
    }
    seen.into_iter().all(|is_seen| is_seen)
}

fn driver_wall(driver: &crate::product_design::ProductDriver) -> Option<InteriorWall> {
    const TOLERANCE: f64 = 1.0e-9;
    let [x, y, z] = driver.normal;
    if (x - 1.0).abs() <= TOLERANCE && y.abs() <= TOLERANCE && z.abs() <= TOLERANCE {
        Some(InteriorWall::Right)
    } else if (x + 1.0).abs() <= TOLERANCE && y.abs() <= TOLERANCE && z.abs() <= TOLERANCE {
        Some(InteriorWall::Left)
    } else if (y - 1.0).abs() <= TOLERANCE && x.abs() <= TOLERANCE && z.abs() <= TOLERANCE {
        Some(InteriorWall::Top)
    } else if (y + 1.0).abs() <= TOLERANCE && x.abs() <= TOLERANCE && z.abs() <= TOLERANCE {
        Some(InteriorWall::Bottom)
    } else if (z - 1.0).abs() <= TOLERANCE && x.abs() <= TOLERANCE && y.abs() <= TOLERANCE {
        Some(InteriorWall::Front)
    } else if (z + 1.0).abs() <= TOLERANCE && x.abs() <= TOLERANCE && y.abs() <= TOLERANCE {
        Some(InteriorWall::Back)
    } else {
        None
    }
}

fn body_fitted_wall_for_face(face: DriverMountingFace) -> InteriorWall {
    match face {
        DriverMountingFace::Front => InteriorWall::Front,
        DriverMountingFace::Back => InteriorWall::Back,
        DriverMountingFace::Right => InteriorWall::Right,
        DriverMountingFace::Left => InteriorWall::Left,
        DriverMountingFace::Top => InteriorWall::Top,
        DriverMountingFace::Bottom => InteriorWall::Bottom,
    }
}

fn outer_wall_at(cabinet: &CabinetDesign, point: [f64; 3]) -> Option<InteriorWall> {
    let [x, y, z] = point;
    if z.abs() <= FRONT_PLANE_TOLERANCE_M {
        Some(InteriorWall::Front)
    } else if (z + cabinet.box_depth_m).abs() <= FRONT_PLANE_TOLERANCE_M {
        Some(InteriorWall::Back)
    } else if (x + cabinet.box_width_m * 0.5).abs() <= FRONT_PLANE_TOLERANCE_M {
        Some(InteriorWall::Left)
    } else if (x - cabinet.box_width_m * 0.5).abs() <= FRONT_PLANE_TOLERANCE_M {
        Some(InteriorWall::Right)
    } else if (y + cabinet.box_height_m * 0.5).abs() <= FRONT_PLANE_TOLERANCE_M {
        Some(InteriorWall::Bottom)
    } else if (y - cabinet.box_height_m * 0.5).abs() <= FRONT_PLANE_TOLERANCE_M {
        Some(InteriorWall::Top)
    } else {
        None
    }
}

fn driver_at(
    design: &ProductDesign,
    wall: InteriorWall,
    point: [f64; 3],
) -> Option<ProductInteriorPatchKind> {
    design
        .drivers
        .iter()
        .enumerate()
        .find_map(|(index, driver)| {
            if driver_wall(driver) != Some(wall) {
                return None;
            }
            let (first, second) = match wall {
                InteriorWall::Front | InteriorWall::Back => (0, 1),
                InteriorWall::Left | InteriorWall::Right => (1, 2),
                InteriorWall::Bottom | InteriorWall::Top => (0, 2),
            };
            let dx = point[first] - driver.position_m[first];
            let dy = point[second] - driver.position_m[second];
            (dx * dx + dy * dy <= driver.radius_m * driver.radius_m).then_some(
                ProductInteriorPatchKind::DriverRear {
                    driver_index: index,
                },
            )
        })
}

fn canonical_patch_normal(
    design: &ProductDesign,
    kind: ProductInteriorPatchKind,
) -> Option<[f64; 3]> {
    match kind {
        ProductInteriorPatchKind::RigidWall(wall) => Some(wall.outward_normal()),
        ProductInteriorPatchKind::DriverRear { driver_index } => {
            design.drivers.get(driver_index).map(|driver| driver.normal)
        }
        ProductInteriorPatchKind::PortMouth => {
            design.cabinet.port.as_ref().map(|_| [0.0, 0.0, 1.0])
        }
        ProductInteriorPatchKind::WaveguideWall { .. }
        | ProductInteriorPatchKind::PortWall
        | ProductInteriorPatchKind::BraceWall { .. } => None,
    }
}

fn marker_for(kind: ProductInteriorPatchKind) -> Result<i32, ProductInteriorMeshError> {
    match kind {
        ProductInteriorPatchKind::RigidWall(wall) => Ok(wall.marker()),
        ProductInteriorPatchKind::DriverRear { driver_index } => {
            let index = i32::try_from(driver_index)
                .map_err(|_| ProductInteriorMeshError::InvalidMeshConfig("too many drivers"))?;
            DRIVER_REAR_MARKER_BASE.checked_add(index).ok_or(
                ProductInteriorMeshError::InvalidMeshConfig("too many drivers"),
            )
        }
        ProductInteriorPatchKind::WaveguideWall { driver_index } => {
            let index = i32::try_from(driver_index)
                .map_err(|_| ProductInteriorMeshError::InvalidMeshConfig("too many drivers"))?;
            WAVEGUIDE_WALL_MARKER_BASE.checked_add(index).ok_or(
                ProductInteriorMeshError::InvalidMeshConfig("too many drivers"),
            )
        }
        ProductInteriorPatchKind::PortMouth => Ok(PORT_MOUTH_MARKER),
        ProductInteriorPatchKind::PortWall => Ok(PORT_WALL_MARKER),
        ProductInteriorPatchKind::BraceWall { brace_index } => {
            let index = i32::try_from(brace_index)
                .map_err(|_| ProductInteriorMeshError::InvalidMeshConfig("too many braces"))?;
            BRACE_WALL_MARKER_BASE.checked_add(index).ok_or(
                ProductInteriorMeshError::InvalidMeshConfig("too many braces"),
            )
        }
    }
}

fn boundary_centroid(
    mesh: &FemMesh,
    face_index: usize,
) -> Result<(f64, f64, f64), ProductInteriorMeshError> {
    let face = mesh
        .boundaries
        .get(face_index)
        .ok_or(ProductInteriorMeshError::InvalidBoundaryFace { face: face_index })?;
    if face.nodes.len() != 3 {
        return Err(ProductInteriorMeshError::InvalidBoundaryFace { face: face_index });
    }
    let points = [
        mesh.nodes
            .get(face.nodes[0])
            .ok_or(ProductInteriorMeshError::InvalidBoundaryFace { face: face_index })?,
        mesh.nodes
            .get(face.nodes[1])
            .ok_or(ProductInteriorMeshError::InvalidBoundaryFace { face: face_index })?,
        mesh.nodes
            .get(face.nodes[2])
            .ok_or(ProductInteriorMeshError::InvalidBoundaryFace { face: face_index })?,
    ];
    Ok((
        (points[0].x + points[1].x + points[2].x) / 3.0,
        (points[0].y + points[1].y + points[2].y) / 3.0,
        (points[0].z + points[1].z + points[2].z) / 3.0,
    ))
}

fn boundary_area(mesh: &FemMesh, face_index: usize) -> Result<f64, ProductInteriorMeshError> {
    let face = mesh
        .boundaries
        .get(face_index)
        .ok_or(ProductInteriorMeshError::InvalidBoundaryFace { face: face_index })?;
    if face.nodes.len() != 3 {
        return Err(ProductInteriorMeshError::InvalidBoundaryFace { face: face_index });
    }
    let p0 = &mesh.nodes[face.nodes[0]];
    let p1 = &mesh.nodes[face.nodes[1]];
    let p2 = &mesh.nodes[face.nodes[2]];
    let ax = p1.x - p0.x;
    let ay = p1.y - p0.y;
    let az = p1.z - p0.z;
    let bx = p2.x - p0.x;
    let by = p2.y - p0.y;
    let bz = p2.z - p0.z;
    let cx = ay * bz - az * by;
    let cy = az * bx - ax * bz;
    let cz = ax * by - ay * bx;
    Ok(0.5 * (cx * cx + cy * cy + cz * cz).sqrt())
}

fn tetrahedral_volume(mesh: &FemMesh) -> f64 {
    mesh.elements
        .iter()
        .filter(|element| element.nodes.len() == 4)
        .map(|element| {
            let p0 = &mesh.nodes[element.nodes[0]];
            let p1 = &mesh.nodes[element.nodes[1]];
            let p2 = &mesh.nodes[element.nodes[2]];
            let p3 = &mesh.nodes[element.nodes[3]];
            let ax = p1.x - p0.x;
            let ay = p1.y - p0.y;
            let az = p1.z - p0.z;
            let bx = p2.x - p0.x;
            let by = p2.y - p0.y;
            let bz = p2.z - p0.z;
            let cx = p3.x - p0.x;
            let cy = p3.y - p0.y;
            let cz = p3.z - p0.z;
            (ax * (by * cz - bz * cy) - ay * (bx * cz - bz * cx) + az * (bx * cy - by * cx)).abs()
                / 6.0
        })
        .sum()
}

fn patch_name(kind: ProductInteriorPatchKind) -> String {
    match kind {
        ProductInteriorPatchKind::RigidWall(wall) => format!("rigid_wall_{wall:?}"),
        ProductInteriorPatchKind::DriverRear { driver_index } => {
            format!("driver_rear_{driver_index}")
        }
        ProductInteriorPatchKind::WaveguideWall { driver_index } => {
            format!("waveguide_wall_{driver_index}")
        }
        ProductInteriorPatchKind::PortMouth => "port_mouth".to_owned(),
        ProductInteriorPatchKind::PortWall => "port_wall".to_owned(),
        ProductInteriorPatchKind::BraceWall { brace_index } => {
            format!("brace_wall_{brace_index}")
        }
    }
}

fn geometry_fingerprint(design: &ProductDesign, config: ProductInteriorMeshConfig) -> String {
    let mut hash = Fnv1a64::default();
    hash.write_u64(MARKER_VERSION);
    for value in [
        design.cabinet.box_width_m,
        design.cabinet.box_height_m,
        design.cabinet.box_depth_m,
        config.max_element_size_m,
    ] {
        hash.write_f64(value);
    }
    hash.write_u64(config.min_cells_per_axis as u64);
    hash.write_u64(config.max_cells_per_axis as u64);
    hash.write_u64(design.drivers.len() as u64);
    for driver in &design.drivers {
        hash.write_str(&driver.id);
        hash.write_f64(driver.radius_m);
        for value in driver.position_m.into_iter().chain(driver.normal) {
            hash.write_f64(value);
        }
    }
    match design.cabinet.port {
        Some(port) => {
            // Cache identity must include every quantity that changes the
            // future body-fitted fluid/solid topology, including an omitted
            // wall thickness from a legacy artifact.
            hash.write_u64(1);
            hash.write_f64(port.diameter_m);
            hash.write_f64(port.length_m);
            hash.write_f64(port.position_m[0]);
            hash.write_f64(port.position_m[1]);
            match port.wall_thickness_m {
                Some(thickness) => {
                    hash.write_u64(1);
                    hash.write_f64(thickness);
                }
                None => hash.write_u64(0),
            }
        }
        None => hash.write_u64(0),
    }
    hash.write_u64(design.cabinet.braces.len() as u64);
    for brace in &design.cabinet.braces {
        hash.write_u64(brace.panel as u64);
        hash.write_u64(brace.split_along as u64);
        hash.write_f64(brace.position_fraction);
        match brace.cross_section {
            Some(cross_section) => {
                hash.write_u64(1);
                hash.write_f64(cross_section.width_m);
                hash.write_f64(cross_section.depth_m);
            }
            None => hash.write_u64(0),
        }
    }
    match &design.waveguide {
        Some(waveguide) => {
            hash.write_u64(1);
            hash.write_str(&waveguide.driver_id);
            hash.write_f64(waveguide.depth_m);
            hash.write_f64(waveguide.surface_specific_impedance_pa_s_per_m);
            hash.write_u64(waveguide.horizontal_spline.len() as u64);
            for [parameter, radius] in &waveguide.horizontal_spline {
                hash.write_f64(*parameter);
                hash.write_f64(*radius);
            }
            hash.write_u64(waveguide.vertical_spline.len() as u64);
            for [parameter, radius] in &waveguide.vertical_spline {
                hash.write_f64(*parameter);
                hash.write_f64(*radius);
            }
        }
        None => hash.write_u64(0),
    }
    format!("product-interior-v{MARKER_VERSION}-{:016x}", hash.finish())
}

#[derive(Clone, Copy)]
struct Fnv1a64(u64);

impl Default for Fnv1a64 {
    fn default() -> Self {
        Self(0xcbf2_9ce4_8422_2325)
    }
}

impl Fnv1a64 {
    fn write_bytes(&mut self, bytes: &[u8]) {
        for &byte in bytes {
            self.0 ^= u64::from(byte);
            self.0 = self.0.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }

    fn write_u64(&mut self, value: u64) {
        self.write_bytes(&value.to_le_bytes());
    }

    fn write_f64(&mut self, value: f64) {
        self.write_u64(value.to_bits());
    }

    fn write_str(&mut self, value: &str) {
        self.write_u64(value.len() as u64);
        self.write_bytes(value.as_bytes());
    }

    fn finish(self) -> u64 {
        self.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cabinet_design::{
        BraceCrossSection, BraceDesign, CabinetDesign, PanelAxis, PanelId,
    };
    use crate::product_design::{CrossoverBranch, CrossoverNetwork, ProductDriver};

    fn sealed_product() -> ProductDesign {
        ProductDesign {
            cabinet: CabinetDesign {
                port: None,
                ..CabinetDesign::default()
            },
            drivers: vec![ProductDriver {
                id: "woofer".into(),
                radius_m: 0.045,
                position_m: [0.0, 0.0, 0.0],
                normal: [0.0, 0.0, 1.0],
                force_factor_tm: 6.0,
                coil_resistance_ohm: 6.0,
                coil_reactance_ohm: 0.2,
                mechanical_resistance_n_s_per_m: 1.0,
                mechanical_reactance_n_s_per_m: 0.1,
                crossover_branch: 0,
            }],
            crossover: CrossoverNetwork {
                branches: vec![CrossoverBranch::default()],
            },
            waveguide: None,
            structural_acoustic: None,
        }
    }

    fn assert_watertight_boundary(mesh: &FemMesh) {
        let mut faces = std::collections::BTreeSet::new();
        let mut edges = BTreeMap::<[usize; 2], usize>::new();
        for boundary in &mesh.boundaries {
            assert_eq!(boundary.nodes.len(), 3);
            let mut face = [boundary.nodes[0], boundary.nodes[1], boundary.nodes[2]];
            face.sort_unstable();
            assert!(faces.insert(face), "duplicate boundary face {face:?}");
            for [left, right] in [
                [boundary.nodes[0], boundary.nodes[1]],
                [boundary.nodes[1], boundary.nodes[2]],
                [boundary.nodes[2], boundary.nodes[0]],
            ] {
                let edge = if left < right {
                    [left, right]
                } else {
                    [right, left]
                };
                *edges.entry(edge).or_default() += 1;
            }
        }
        assert!(
            edges.values().all(|&uses| uses == 2),
            "boundary has non-manifold edges: {:?}",
            edges
                .iter()
                .filter(|&(_, &uses)| uses != 2)
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn sealed_mesh_has_complete_stable_patch_registry_and_volume() {
        let mesh = build_sealed_product_interior_mesh(
            &sealed_product(),
            ProductInteriorMeshConfig {
                max_element_size_m: 0.04,
                ..ProductInteriorMeshConfig::default()
            },
        )
        .expect("sealed product mesh");

        assert_eq!(mesh.patches.len(), 7);
        assert!(
            mesh.patch(ProductInteriorPatchKind::DriverRear { driver_index: 0 })
                .is_some_and(|patch| patch.area_m2 > 0.0)
        );
        for wall in InteriorWall::ALL {
            assert!(
                mesh.patch(ProductInteriorPatchKind::RigidWall(wall))
                    .is_some_and(|patch| patch.area_m2 > 0.0)
            );
        }
        assert!(
            (mesh.diagnostics.fluid_volume_m3 - mesh.diagnostics.expected_box_volume_m3).abs()
                < 1.0e-12
        );
        assert_eq!(
            mesh.mesh.boundaries.len(),
            mesh.patches
                .iter()
                .map(|patch| patch.boundary_faces.len())
                .sum::<usize>()
        );
    }

    #[test]
    fn body_fitted_sealed_mesh_preserves_circular_driver_rim_and_wall_markers() {
        let product = sealed_product();
        let mesh = build_body_fitted_sealed_product_interior_mesh(
            &product,
            ProductInteriorMeshConfig::default(),
        )
        .expect("body-fitted sealed product mesh");

        let driver = mesh
            .patch(ProductInteriorPatchKind::DriverRear { driver_index: 0 })
            .expect("body-fitted driver patch");
        let expected_driver_area = std::f64::consts::PI * product.drivers[0].radius_m.powi(2);
        assert!(
            (driver.area_m2 - expected_driver_area).abs() / expected_driver_area < 0.02,
            "driver area {} must retain the circular surface area {}",
            driver.area_m2,
            expected_driver_area
        );
        for wall in InteriorWall::ALL {
            assert!(
                mesh.patch(ProductInteriorPatchKind::RigidWall(wall))
                    .is_some_and(|patch| patch.area_m2 > 0.0),
                "missing body-fitted {wall:?} patch"
            );
        }
        assert!(
            (mesh.diagnostics.fluid_volume_m3 - product.cabinet.volume_m3()).abs()
                / product.cabinet.volume_m3()
                < 1.0e-10
        );
        assert!(mesh.geometry_fingerprint.contains("body-fitted-driver"));
    }

    #[test]
    fn body_fitted_mesh_supports_every_axis_aligned_driver_wall() {
        let base = sealed_product();
        let w = base.cabinet.box_width_m;
        let h = base.cabinet.box_height_m;
        let d = base.cabinet.box_depth_m;
        let cases = [
            ([0.0, 0.0, 1.0], [0.0, 0.0, 0.0], InteriorWall::Front),
            ([0.0, 0.0, -1.0], [0.0, 0.0, -d], InteriorWall::Back),
            (
                [1.0, 0.0, 0.0],
                [w * 0.5, 0.0, -d * 0.5],
                InteriorWall::Right,
            ),
            (
                [-1.0, 0.0, 0.0],
                [-w * 0.5, 0.0, -d * 0.5],
                InteriorWall::Left,
            ),
            ([0.0, 1.0, 0.0], [0.0, h * 0.5, -d * 0.5], InteriorWall::Top),
            (
                [0.0, -1.0, 0.0],
                [0.0, -h * 0.5, -d * 0.5],
                InteriorWall::Bottom,
            ),
        ];

        for (normal, position_m, wall) in cases {
            let mut design = base.clone();
            design.drivers[0].normal = normal;
            design.drivers[0].position_m = position_m;
            let mesh = build_body_fitted_sealed_product_interior_mesh(
                &design,
                ProductInteriorMeshConfig::default(),
            )
            .expect("axis-aligned body-fitted mesh");
            let driver = mesh
                .patch(ProductInteriorPatchKind::DriverRear { driver_index: 0 })
                .expect("body-fitted driver patch");
            assert_eq!(
                driver.canonical_normal_m,
                Some(normal),
                "{wall:?} driver normal"
            );
            let expected_area = std::f64::consts::PI * design.drivers[0].radius_m.powi(2);
            let rim_radii = driver
                .boundary_faces
                .iter()
                .flat_map(|&face_index| mesh.mesh.boundaries[face_index].nodes.iter())
                .map(|&node| {
                    let point = &mesh.mesh.nodes[node];
                    ((point.x - position_m[0]).powi(2)
                        + (point.y - position_m[1]).powi(2)
                        + (point.z - position_m[2]).powi(2))
                    .sqrt()
                })
                .filter(|&radius| radius > 1.0e-12)
                .collect::<Vec<_>>();
            assert!(
                rim_radii
                    .iter()
                    .all(|&radius| (radius - design.drivers[0].radius_m).abs() < 1.0e-12),
                "{wall:?} rim radii {rim_radii:?}"
            );
            assert!(
                (driver.area_m2 - expected_area).abs() / expected_area < 0.02,
                "{wall:?} driver rim area {} ({} faces; {}) must retain circular area {expected_area}",
                driver.area_m2,
                driver.boundary_faces.len(),
                mesh.geometry_fingerprint
            );
            for &face_index in &driver.boundary_faces {
                let (x, y, z) = boundary_centroid(&mesh.mesh, face_index).expect("face centroid");
                let on_driver_wall = match wall {
                    InteriorWall::Front => z.abs() <= FRONT_PLANE_TOLERANCE_M,
                    InteriorWall::Back => (z + d).abs() <= FRONT_PLANE_TOLERANCE_M,
                    InteriorWall::Left => (x + w * 0.5).abs() <= FRONT_PLANE_TOLERANCE_M,
                    InteriorWall::Right => (x - w * 0.5).abs() <= FRONT_PLANE_TOLERANCE_M,
                    InteriorWall::Bottom => (y + h * 0.5).abs() <= FRONT_PLANE_TOLERANCE_M,
                    InteriorWall::Top => (y - h * 0.5).abs() <= FRONT_PLANE_TOLERANCE_M,
                };
                assert!(on_driver_wall, "driver patch escaped {wall:?}");
            }
            for rigid_wall in InteriorWall::ALL {
                let patch = mesh
                    .patch(ProductInteriorPatchKind::RigidWall(rigid_wall))
                    .expect("rigid wall patch");
                assert!(
                    patch.area_m2 > 0.0,
                    "missing {rigid_wall:?} patch for {wall:?} driver"
                );
                assert_eq!(
                    patch.canonical_normal_m,
                    Some(rigid_wall.outward_normal()),
                    "{rigid_wall:?} patch normal"
                );
            }
            assert!(
                (mesh.diagnostics.fluid_volume_m3 - design.cabinet.volume_m3()).abs()
                    / design.cabinet.volume_m3()
                    < 1.0e-10,
                "{wall:?} mesh volume"
            );
        }
    }

    #[test]
    fn body_fitted_mesh_supports_bounded_tilted_front_driver() {
        let mut product = sealed_product();
        let tilt = 0.3_f64;
        product.drivers[0].normal = [tilt, 0.0, (1.0 - tilt * tilt).sqrt()];

        let mesh = build_body_fitted_sealed_product_interior_mesh(
            &product,
            ProductInteriorMeshConfig::default(),
        )
        .expect("bounded tilted body-fitted mesh");
        let driver = mesh
            .patch(ProductInteriorPatchKind::DriverRear { driver_index: 0 })
            .expect("tilted driver patch");
        assert_eq!(
            driver.canonical_normal_m,
            Some(product.drivers[0].normal),
            "driver patch must retain the declared outward orientation"
        );
        let expected_area = std::f64::consts::PI * product.drivers[0].radius_m.powi(2);
        assert!(
            (driver.area_m2 - expected_area).abs() / expected_area < 0.02,
            "tilted driver area {} must retain the circular surface area {expected_area}",
            driver.area_m2
        );

        let z_coordinates = driver
            .boundary_faces
            .iter()
            .flat_map(|&face_index| mesh.mesh.boundaries[face_index].nodes.iter())
            .map(|&node| mesh.mesh.nodes[node].z)
            .collect::<Vec<_>>();
        let z_min = z_coordinates.iter().copied().fold(f64::INFINITY, f64::min);
        let z_max = z_coordinates
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);
        assert!(
            z_max - z_min > 1.0e-4,
            "tilted driver patch must not be coplanar with the front wall: z=[{z_min}, {z_max}]"
        );
        assert!(
            mesh.patch(ProductInteriorPatchKind::RigidWall(InteriorWall::Front))
                .is_some_and(|patch| patch.area_m2 > 0.0)
        );
        assert!(mesh.diagnostics.fluid_volume_m3.is_finite());
        assert!(mesh.diagnostics.fluid_volume_m3 > 0.0);

        assert!(matches!(
            build_sealed_product_interior_mesh(
                &product,
                ProductInteriorMeshConfig {
                    max_element_size_m: 0.04,
                    ..ProductInteriorMeshConfig::default()
                },
            ),
            Err(ProductInteriorMeshError::BodyFittedDriverLayoutUnsupported)
        ));
    }

    #[test]
    fn body_fitted_mesh_supports_bounded_tilt_on_every_cabinet_face() {
        let base = sealed_product();
        let w = base.cabinet.box_width_m;
        let h = base.cabinet.box_height_m;
        let d = base.cabinet.box_depth_m;
        let tilt = 0.3_f64;
        let normal_axis = (1.0 - tilt * tilt).sqrt();
        let cases = [
            (
                DriverMountingFace::Front,
                [tilt, 0.0, normal_axis],
                [0.0, 0.0, 0.0],
            ),
            (
                DriverMountingFace::Back,
                [tilt, 0.0, -normal_axis],
                [0.0, 0.0, -d],
            ),
            (
                DriverMountingFace::Right,
                [normal_axis, 0.0, tilt],
                [w * 0.5, 0.0, -d * 0.5],
            ),
            (
                DriverMountingFace::Left,
                [-normal_axis, 0.0, tilt],
                [-w * 0.5, 0.0, -d * 0.5],
            ),
            (
                DriverMountingFace::Top,
                [0.0, normal_axis, tilt],
                [0.0, h * 0.5, -d * 0.5],
            ),
            (
                DriverMountingFace::Bottom,
                [0.0, -normal_axis, tilt],
                [0.0, -h * 0.5, -d * 0.5],
            ),
        ];

        for (face, normal, wall_center) in cases {
            let mut design = base.clone();
            design.drivers[0].normal = normal;
            design.drivers[0].position_m = wall_center;
            let mesh = build_body_fitted_sealed_product_interior_mesh(
                &design,
                ProductInteriorMeshConfig::default(),
            )
            .unwrap_or_else(|error| panic!("{face:?} tilted mesh: {error}"));
            let patch = mesh
                .patch(ProductInteriorPatchKind::DriverRear { driver_index: 0 })
                .expect("tilted driver patch");
            assert_eq!(patch.canonical_normal_m, Some(normal), "{face:?}");
            let wall_normal = face.outward_normal();
            let wall_distances = patch
                .boundary_faces
                .iter()
                .flat_map(|&face_index| mesh.mesh.boundaries[face_index].nodes.iter())
                .map(|&node| {
                    let point = &mesh.mesh.nodes[node];
                    (point.x - wall_center[0]) * wall_normal[0]
                        + (point.y - wall_center[1]) * wall_normal[1]
                        + (point.z - wall_center[2]) * wall_normal[2]
                })
                .collect::<Vec<_>>();
            let minimum = wall_distances.iter().copied().fold(f64::INFINITY, f64::min);
            let maximum = wall_distances
                .iter()
                .copied()
                .fold(f64::NEG_INFINITY, f64::max);
            assert!(
                maximum - minimum > 1.0e-4,
                "{face:?} driver patch must leave its wall plane"
            );
            assert!(mesh.diagnostics.fluid_volume_m3.is_finite());
            assert!(mesh.diagnostics.fluid_volume_m3 > 0.0);
        }
    }

    #[test]
    fn body_fitted_mesh_supports_drivers_on_distinct_walls() {
        let mut design = sealed_product();
        let mut second = design.drivers[0].clone();
        second.id = "side-tweeter".into();
        second.radius_m = 0.03;
        second.crossover_branch = 1;
        second.normal = [1.0, 0.0, 0.0];
        second.position_m = [
            design.cabinet.box_width_m * 0.5,
            0.0,
            -design.cabinet.box_depth_m * 0.5,
        ];
        design.drivers.push(second);
        design.crossover.branches.push(CrossoverBranch::default());
        let mesh = build_body_fitted_sealed_product_interior_mesh(
            &design,
            ProductInteriorMeshConfig::default(),
        )
        .expect("two-wall body-fitted mesh");
        for (index, driver) in design.drivers.iter().enumerate() {
            let patch = mesh
                .patch(ProductInteriorPatchKind::DriverRear {
                    driver_index: index,
                })
                .expect("driver patch");
            let expected = std::f64::consts::PI * driver.radius_m.powi(2);
            assert!(
                (patch.area_m2 - expected).abs() / expected < 0.02,
                "driver {index} circular area"
            );
        }
        assert_eq!(mesh.patches.len(), 8);
        assert!(
            (mesh.diagnostics.fluid_volume_m3 - design.cabinet.volume_m3()).abs()
                / design.cabinet.volume_m3()
                < 1.0e-10
        );
    }

    #[test]
    fn body_fitted_mesh_supports_tilted_drivers_on_distinct_walls() {
        let mut design = sealed_product();
        let tilt: f64 = 0.2;
        let normal_axis = (1.0 - tilt * tilt).sqrt();
        design.drivers[0].normal = [tilt, 0.0, normal_axis];

        let mut second = design.drivers[0].clone();
        second.id = "tilted-side-tweeter".into();
        second.radius_m = 0.03;
        second.crossover_branch = 1;
        second.normal = [normal_axis, 0.0, tilt];
        second.position_m = [
            design.cabinet.box_width_m * 0.5,
            0.0,
            -design.cabinet.box_depth_m * 0.5,
        ];
        design.drivers.push(second);
        design.crossover.branches.push(CrossoverBranch::default());

        let mesh = build_body_fitted_sealed_product_interior_mesh(
            &design,
            ProductInteriorMeshConfig::default(),
        )
        .expect("two-wall tilted body-fitted mesh");

        for (index, driver) in design.drivers.iter().enumerate() {
            let patch = mesh
                .patch(ProductInteriorPatchKind::DriverRear {
                    driver_index: index,
                })
                .expect("tilted driver patch");
            assert_eq!(patch.canonical_normal_m, Some(driver.normal));
            let expected = std::f64::consts::PI * driver.radius_m.powi(2);
            assert!(
                (patch.area_m2 - expected).abs() / expected < 0.02,
                "driver {index} circular diaphragm area"
            );
            let wall_normal = if index == 0 {
                [0.0, 0.0, 1.0]
            } else {
                [1.0, 0.0, 0.0]
            };
            let wall_center = driver.position_m;
            let distances = patch
                .boundary_faces
                .iter()
                .flat_map(|&face_index| mesh.mesh.boundaries[face_index].nodes.iter())
                .map(|&node_index| {
                    let point = &mesh.mesh.nodes[node_index];
                    (point.x - wall_center[0]) * wall_normal[0]
                        + (point.y - wall_center[1]) * wall_normal[1]
                        + (point.z - wall_center[2]) * wall_normal[2]
                })
                .collect::<Vec<_>>();
            let minimum = distances.iter().copied().fold(f64::INFINITY, f64::min);
            let maximum = distances.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            assert!(maximum - minimum > 1.0e-4, "driver {index} must be tilted");
        }
        assert!(mesh.diagnostics.fluid_volume_m3.is_finite());
        assert!(mesh.diagnostics.fluid_volume_m3 > 0.0);
    }

    #[test]
    fn body_fitted_mesh_supports_two_drivers_on_one_wall() {
        let mut design = sealed_product();
        design.cabinet.box_width_m = 0.4;
        design.cabinet.baffle_width_m = 0.44;
        design.drivers[0].position_m[0] = -0.1;
        let mut second = design.drivers[0].clone();
        second.id = "second-front-driver".into();
        second.crossover_branch = 1;
        second.position_m = [0.1, 0.0, 0.0];
        design.drivers.push(second);
        design.crossover.branches.push(CrossoverBranch::default());
        let mesh = build_body_fitted_sealed_product_interior_mesh(
            &design,
            ProductInteriorMeshConfig::default(),
        )
        .expect("two same-wall drivers mesh");
        for (driver_index, driver) in design.drivers.iter().enumerate() {
            let patch = mesh
                .patch(ProductInteriorPatchKind::DriverRear { driver_index })
                .expect("same-wall driver patch");
            let expected_area = std::f64::consts::PI * driver.radius_m.powi(2);
            assert!((patch.area_m2 - expected_area).abs() / expected_area < 0.02);
        }
        assert!(
            (mesh.diagnostics.fluid_volume_m3 - design.cabinet.volume_m3()).abs()
                / design.cabinet.volume_m3()
                < 1.0e-10
        );
    }

    #[test]
    fn body_fitted_mesh_supports_tilted_drivers_on_one_wall() {
        let mut design = sealed_product();
        design.cabinet.box_width_m = 0.4;
        design.cabinet.baffle_width_m = 0.44;
        let tilt: f64 = 0.2;
        let normal = [tilt, 0.0, (1.0 - tilt * tilt).sqrt()];
        design.drivers[0].position_m[0] = -0.1;
        design.drivers[0].normal = normal;
        let mut second = design.drivers[0].clone();
        second.id = "second-tilted-front-driver".into();
        second.crossover_branch = 1;
        second.position_m = [0.1, 0.0, 0.0];
        design.drivers.push(second);
        design.crossover.branches.push(CrossoverBranch::default());

        let mesh = build_body_fitted_sealed_product_interior_mesh(
            &design,
            ProductInteriorMeshConfig::default(),
        )
        .expect("two same-wall tilted drivers mesh");
        for (driver_index, driver) in design.drivers.iter().enumerate() {
            let patch = mesh
                .patch(ProductInteriorPatchKind::DriverRear { driver_index })
                .expect("same-wall tilted driver patch");
            assert_eq!(patch.canonical_normal_m, Some(driver.normal));
            let expected_area = std::f64::consts::PI * driver.radius_m.powi(2);
            assert!((patch.area_m2 - expected_area).abs() / expected_area < 0.02);
        }
        assert!(mesh.diagnostics.fluid_volume_m3.is_finite());
        assert!(mesh.diagnostics.fluid_volume_m3 > 0.0);
    }

    #[test]
    fn body_fitted_ported_mesh_retains_mouth_and_tube_wall_patches() {
        let mut product = sealed_product();
        product.drivers[0].position_m[0] = -0.06;
        let mut port = CabinetDesign::default().port.expect("default port");
        port.position_m = [0.06, 0.0];
        product.cabinet.port = Some(port);
        let mesh = build_body_fitted_sealed_product_interior_mesh(
            &product,
            ProductInteriorMeshConfig::default(),
        )
        .expect("body-fitted ported mesh");
        assert!(
            mesh.patch(ProductInteriorPatchKind::PortMouth)
                .is_some_and(|patch| patch.area_m2 > 0.0)
        );
        assert!(
            mesh.patch(ProductInteriorPatchKind::PortWall)
                .is_some_and(|patch| patch.area_m2 > 0.0)
        );
        assert_eq!(
            mesh.patch(ProductInteriorPatchKind::PortMouth)
                .expect("port mouth patch")
                .canonical_normal_m,
            Some([0.0, 0.0, 1.0])
        );
        assert_eq!(
            mesh.patch(ProductInteriorPatchKind::PortWall)
                .expect("port wall patch")
                .canonical_normal_m,
            None
        );
        assert!(mesh.diagnostics.fluid_volume_m3 < product.cabinet.volume_m3());
    }

    #[test]
    fn body_fitted_front_port_composes_with_brace_reaching_front_wall() {
        let mut product = sealed_product();
        product.drivers[0].position_m[0] = -0.06;
        let mut port = CabinetDesign::default().port.expect("default port");
        port.position_m = [0.06, 0.0];
        product.cabinet.port = Some(port);
        // This brace spans the full top-panel depth, so its front end meets
        // the port-bearing baffle and must share the front-panel PLC.
        product
            .cabinet
            .braces
            .push(crate::cabinet_design::BraceDesign {
                panel: crate::cabinet_design::PanelId::Top,
                split_along: crate::cabinet_design::PanelAxis::Width,
                position_fraction: 0.5,
                cross_section: Some(crate::cabinet_design::BraceCrossSection {
                    width_m: 0.05,
                    depth_m: 0.05,
                }),
            });

        let mesh = build_body_fitted_sealed_product_interior_mesh(
            &product,
            ProductInteriorMeshConfig {
                max_element_size_m: 0.025,
                ..ProductInteriorMeshConfig::default()
            },
        )
        .expect("front port and non-front spanning brace should compose");

        for kind in [
            ProductInteriorPatchKind::PortMouth,
            ProductInteriorPatchKind::PortWall,
            ProductInteriorPatchKind::BraceWall { brace_index: 0 },
        ] {
            assert!(
                mesh.patch(kind).is_some_and(|patch| patch.area_m2 > 0.0),
                "missing non-empty {kind:?} patch"
            );
        }
        assert!(is_single_fluid_component(&mut mesh.mesh.clone()));
    }

    #[test]
    fn body_fitted_tilted_driver_composes_with_port_and_brace() {
        let mut product = sealed_product();
        let tilt: f64 = 0.2;
        product.drivers[0].position_m[0] = -0.06;
        product.drivers[0].normal = [tilt, 0.0, (1.0 - tilt * tilt).sqrt()];
        let mut port = CabinetDesign::default().port.expect("default port");
        port.position_m = [0.06, 0.0];
        product.cabinet.port = Some(port);
        product
            .cabinet
            .braces
            .push(crate::cabinet_design::BraceDesign {
                panel: crate::cabinet_design::PanelId::Top,
                split_along: crate::cabinet_design::PanelAxis::Width,
                position_fraction: 0.5,
                cross_section: Some(crate::cabinet_design::BraceCrossSection {
                    width_m: 0.05,
                    depth_m: 0.05,
                }),
            });

        let mesh = build_body_fitted_sealed_product_interior_mesh(
            &product,
            ProductInteriorMeshConfig {
                max_element_size_m: 0.025,
                ..ProductInteriorMeshConfig::default()
            },
        )
        .expect("tilted driver, port, and brace should compose");
        assert_watertight_boundary(&mesh.mesh);
        let driver = mesh
            .patch(ProductInteriorPatchKind::DriverRear { driver_index: 0 })
            .expect("tilted driver patch");
        assert_eq!(driver.canonical_normal_m, Some(product.drivers[0].normal));
        let expected_area = std::f64::consts::PI * product.drivers[0].radius_m.powi(2);
        assert!((driver.area_m2 - expected_area).abs() / expected_area < 0.02);
        for kind in [
            ProductInteriorPatchKind::PortMouth,
            ProductInteriorPatchKind::PortWall,
            ProductInteriorPatchKind::BraceWall { brace_index: 0 },
        ] {
            assert!(
                mesh.patch(kind).is_some_and(|patch| patch.area_m2 > 0.0),
                "missing non-empty {kind:?} patch"
            );
        }
        assert!(is_single_fluid_component(&mut mesh.mesh.clone()));
        assert!(mesh.diagnostics.fluid_volume_m3.is_finite());
        assert!(mesh.diagnostics.fluid_volume_m3 > 0.0);
    }

    #[test]
    fn body_fitted_port_and_intersecting_brace_fail_closed() {
        let mut product = sealed_product();
        product.drivers[0].position_m[0] = -0.08;
        product.drivers[0].radius_m = 0.03;
        let mut port = CabinetDesign::default().port.expect("default port");
        port.position_m = [0.0, 0.0];
        port.length_m = product.cabinet.box_depth_m;
        product.cabinet.port = Some(port);
        product
            .cabinet
            .braces
            .push(crate::cabinet_design::BraceDesign {
                panel: crate::cabinet_design::PanelId::Back,
                split_along: crate::cabinet_design::PanelAxis::Width,
                position_fraction: 0.5,
                cross_section: Some(crate::cabinet_design::BraceCrossSection {
                    width_m: 0.05,
                    depth_m: 0.05,
                }),
            });

        assert!(matches!(
            build_body_fitted_sealed_product_interior_mesh(
                &product,
                ProductInteriorMeshConfig {
                    max_element_size_m: 0.025,
                    ..ProductInteriorMeshConfig::default()
                },
            ),
            Err(ProductInteriorMeshError::PortBraceIntersection { brace_index: 0 })
        ));
    }

    #[test]
    fn body_fitted_rim_refines_with_requested_element_size() {
        let product = sealed_product();
        let coarse = build_body_fitted_sealed_product_interior_mesh(
            &product,
            ProductInteriorMeshConfig {
                max_element_size_m: 0.025,
                ..ProductInteriorMeshConfig::default()
            },
        )
        .expect("coarse body-fitted mesh");
        let fine = build_body_fitted_sealed_product_interior_mesh(
            &product,
            ProductInteriorMeshConfig {
                max_element_size_m: 0.006,
                ..ProductInteriorMeshConfig::default()
            },
        )
        .expect("fine body-fitted mesh");

        assert!(coarse.geometry_fingerprint.ends_with("-24"));
        assert!(fine.geometry_fingerprint.ends_with("-48"));
        assert!(fine.diagnostics.tetrahedron_count > coarse.diagnostics.tetrahedron_count);

        let exact_area = std::f64::consts::PI * product.drivers[0].radius_m.powi(2);
        let area_error = |mesh: &ProductInteriorMesh| {
            (mesh
                .patch(ProductInteriorPatchKind::DriverRear { driver_index: 0 })
                .expect("driver patch")
                .area_m2
                - exact_area)
                .abs()
        };
        assert!(
            area_error(&fine) < area_error(&coarse),
            "finer rim must better approximate the exact circular driver area"
        );

        let repeat = build_body_fitted_sealed_product_interior_mesh(
            &product,
            ProductInteriorMeshConfig {
                max_element_size_m: 0.006,
                ..ProductInteriorMeshConfig::default()
            },
        )
        .expect("repeat fine body-fitted mesh");
        assert_eq!(fine.geometry_fingerprint, repeat.geometry_fingerprint);
        assert_eq!(fine.diagnostics, repeat.diagnostics);
    }

    #[test]
    fn body_fitted_rim_respects_resolution_limit() {
        let error = build_body_fitted_sealed_product_interior_mesh(
            &sealed_product(),
            ProductInteriorMeshConfig {
                max_element_size_m: 0.001,
                min_cells_per_axis: 2,
                max_cells_per_axis: 8,
            },
        )
        .expect_err("excessively fine rim must obey the resource limit");
        assert!(matches!(
            error,
            ProductInteriorMeshError::BodyFittedResolutionLimit {
                requested: 284,
                maximum: 32,
            }
        ));
    }

    #[test]
    fn body_fitted_brace_wall_uses_the_thinnest_contact_axis() {
        let product = sealed_product();
        let cabinet = &product.cabinet;
        let back_spanning_brace = AxisAlignedSolid {
            min_m: [-cabinet.box_width_m * 0.5, -0.025, -cabinet.box_depth_m],
            max_m: [
                cabinet.box_width_m * 0.5,
                0.025,
                -cabinet.box_depth_m + 0.05,
            ],
        };
        assert_eq!(
            body_fitted_brace_wall(back_spanning_brace, cabinet),
            Some(InteriorWall::Back)
        );

        let right_spanning_brace = AxisAlignedSolid {
            min_m: [
                cabinet.box_width_m * 0.5 - 0.05,
                -cabinet.box_height_m * 0.5,
                -cabinet.box_depth_m,
            ],
            max_m: [cabinet.box_width_m * 0.5, cabinet.box_height_m * 0.5, 0.0],
        };
        assert_eq!(
            body_fitted_brace_wall(right_spanning_brace, cabinet),
            Some(InteriorWall::Right)
        );
    }

    #[test]
    fn side_wall_driver_receives_its_own_typed_patch() {
        let mut design = sealed_product();
        design.drivers[0].normal = [1.0, 0.0, 0.0];
        design.drivers[0].position_m = [
            design.cabinet.box_width_m * 0.5,
            0.0,
            -design.cabinet.box_depth_m * 0.5,
        ];
        let mesh = build_sealed_product_interior_mesh(
            &design,
            ProductInteriorMeshConfig {
                max_element_size_m: 0.04,
                ..ProductInteriorMeshConfig::default()
            },
        )
        .expect("right-wall driver mesh");

        let patch = mesh
            .patch(ProductInteriorPatchKind::DriverRear { driver_index: 0 })
            .expect("driver patch");
        assert!(patch.area_m2 > 0.0);
        for &face_index in &patch.boundary_faces {
            let (x, _, _) = boundary_centroid(&mesh.mesh, face_index).expect("face centroid");
            assert!((x - design.cabinet.box_width_m * 0.5).abs() <= FRONT_PLANE_TOLERANCE_M);
        }
    }

    #[test]
    fn every_axis_aligned_wall_accepts_and_marks_a_driver_patch() {
        let base = sealed_product();
        let w = base.cabinet.box_width_m;
        let h = base.cabinet.box_height_m;
        let d = base.cabinet.box_depth_m;
        let cases = [
            ([0.0, 0.0, 1.0], [0.0, 0.0, 0.0], InteriorWall::Front),
            ([0.0, 0.0, -1.0], [0.0, 0.0, -d], InteriorWall::Back),
            (
                [1.0, 0.0, 0.0],
                [w * 0.5, 0.0, -d * 0.5],
                InteriorWall::Right,
            ),
            (
                [-1.0, 0.0, 0.0],
                [-w * 0.5, 0.0, -d * 0.5],
                InteriorWall::Left,
            ),
            ([0.0, 1.0, 0.0], [0.0, h * 0.5, -d * 0.5], InteriorWall::Top),
            (
                [0.0, -1.0, 0.0],
                [0.0, -h * 0.5, -d * 0.5],
                InteriorWall::Bottom,
            ),
        ];

        for (normal, position_m, wall) in cases {
            let mut design = base.clone();
            design.drivers[0].normal = normal;
            design.drivers[0].position_m = position_m;
            let mesh = build_sealed_product_interior_mesh(
                &design,
                ProductInteriorMeshConfig {
                    max_element_size_m: 0.04,
                    ..ProductInteriorMeshConfig::default()
                },
            )
            .expect("axis-aligned driver mesh");
            let patch = mesh
                .patch(ProductInteriorPatchKind::DriverRear { driver_index: 0 })
                .expect("driver patch");
            assert!(patch.area_m2 > 0.0, "{wall:?}");
            for &face_index in &patch.boundary_faces {
                let (x, y, z) = boundary_centroid(&mesh.mesh, face_index).expect("face centroid");
                let on_wall = match wall {
                    InteriorWall::Front => z.abs() <= FRONT_PLANE_TOLERANCE_M,
                    InteriorWall::Back => (z + d).abs() <= FRONT_PLANE_TOLERANCE_M,
                    InteriorWall::Left => (x + w * 0.5).abs() <= FRONT_PLANE_TOLERANCE_M,
                    InteriorWall::Right => (x - w * 0.5).abs() <= FRONT_PLANE_TOLERANCE_M,
                    InteriorWall::Bottom => (y + h * 0.5).abs() <= FRONT_PLANE_TOLERANCE_M,
                    InteriorWall::Top => (y - h * 0.5).abs() <= FRONT_PLANE_TOLERANCE_M,
                };
                assert!(on_wall, "driver patch escaped {wall:?}");
            }
        }
    }

    #[test]
    fn mesh_fingerprint_is_deterministic_and_geometry_sensitive() {
        let config = ProductInteriorMeshConfig {
            max_element_size_m: 0.04,
            ..ProductInteriorMeshConfig::default()
        };
        let first = build_sealed_product_interior_mesh(&sealed_product(), config).unwrap();
        let second = build_sealed_product_interior_mesh(&sealed_product(), config).unwrap();
        assert_eq!(first.geometry_fingerprint, second.geometry_fingerprint);

        let mut moved = sealed_product();
        moved.drivers[0].position_m[0] = 0.01;
        let moved = build_sealed_product_interior_mesh(&moved, config).unwrap();
        assert_ne!(first.geometry_fingerprint, moved.geometry_fingerprint);

        let mut waveguide = sealed_product();
        waveguide.waveguide = Some(crate::ProductWaveguideDesign {
            driver_id: "woofer".into(),
            horizontal_spline: vec![[0.0, 0.06], [0.5, 0.05], [1.0, 0.045]],
            vertical_spline: vec![[0.0, 0.06], [0.5, 0.05], [1.0, 0.045]],
            depth_m: 0.02,
            surface_specific_impedance_pa_s_per_m: 2.5e6,
        });
        let waveguide_mesh = build_sealed_product_interior_mesh(&waveguide, config).unwrap();
        assert_ne!(
            first.geometry_fingerprint, waveguide_mesh.geometry_fingerprint,
            "waveguide geometry must participate in cache identity"
        );
    }

    #[test]
    fn body_fitted_waveguide_lowers_spline_rings_and_typed_surface() {
        let mut design = sealed_product();
        design.waveguide = Some(crate::ProductWaveguideDesign {
            driver_id: "woofer".into(),
            horizontal_spline: vec![[0.0, 0.06], [0.4, 0.052], [1.0, 0.045]],
            vertical_spline: vec![[0.0, 0.055], [0.7, 0.048], [1.0, 0.045]],
            depth_m: 0.02,
            surface_specific_impedance_pa_s_per_m: 2.5e6,
        });
        let mesh = build_body_fitted_sealed_product_interior_mesh(
            &design,
            ProductInteriorMeshConfig::default(),
        )
        .expect("body-fitted waveguide mesh");

        let waveguide = mesh
            .patch(ProductInteriorPatchKind::WaveguideWall { driver_index: 0 })
            .expect("waveguide wall patch");
        assert!(waveguide.area_m2 > 0.0);
        assert_eq!(mesh.patches.len(), InteriorWall::ALL.len() + 2);
        assert_watertight_boundary(&mesh.mesh);

        let guide_z = waveguide
            .boundary_faces
            .iter()
            .flat_map(|&face_index| mesh.mesh.boundaries[face_index].nodes.iter())
            .map(|&node| mesh.mesh.nodes[node].z)
            .collect::<Vec<_>>();
        assert!(guide_z.iter().copied().any(|z| z.abs() < 1.0e-12));
        assert!(guide_z.iter().copied().any(|z| z < -0.019));

        let driver = mesh
            .patch(ProductInteriorPatchKind::DriverRear { driver_index: 0 })
            .expect("waveguide throat driver patch");
        assert!(driver.area_m2 > 0.0);
        assert!(
            driver
                .boundary_faces
                .iter()
                .flat_map(|&face_index| mesh.mesh.boundaries[face_index].nodes.iter())
                .map(|&node| mesh.mesh.nodes[node].z)
                .all(|z| (z + 0.02).abs() < 1.0e-12)
        );
    }

    #[test]
    fn unresolved_port_and_brace_geometry_fail_closed() {
        let mut ported = sealed_product();
        ported.drivers[0].position_m[0] = -0.06;
        ported.cabinet.port = CabinetDesign::default().port;
        ported.cabinet.port.as_mut().unwrap().position_m = [0.06, 0.0];
        assert!(matches!(
            build_sealed_product_interior_mesh(&ported, ProductInteriorMeshConfig::default()),
            Err(ProductInteriorMeshError::ResolvedPortUnsupported)
        ));
        ported.cabinet.port.as_mut().unwrap().wall_thickness_m = None;
        assert!(matches!(
            build_sealed_product_interior_mesh(&ported, ProductInteriorMeshConfig::default()),
            Err(ProductInteriorMeshError::MissingPortWallGeometry)
        ));

        let mut braced = sealed_product();
        braced
            .cabinet
            .braces
            .push(crate::cabinet_design::BraceDesign {
                panel: crate::cabinet_design::PanelId::Back,
                split_along: crate::cabinet_design::PanelAxis::Width,
                position_fraction: 0.5,
                cross_section: None,
            });
        assert!(matches!(
            build_sealed_product_interior_mesh(&braced, ProductInteriorMeshConfig::default()),
            Err(ProductInteriorMeshError::MissingBraceCrossSection)
        ));
    }

    #[test]
    fn resolved_panel_rib_is_subtracted_and_has_a_typed_wall_patch() {
        let mut product = sealed_product();
        product
            .cabinet
            .braces
            .push(crate::cabinet_design::BraceDesign {
                panel: crate::cabinet_design::PanelId::Top,
                split_along: crate::cabinet_design::PanelAxis::Width,
                position_fraction: 0.5,
                cross_section: Some(crate::cabinet_design::BraceCrossSection {
                    width_m: 0.05,
                    depth_m: 0.05,
                }),
            });
        let config = ProductInteriorMeshConfig {
            max_element_size_m: 0.025,
            ..ProductInteriorMeshConfig::default()
        };
        let unbraced = build_sealed_product_interior_mesh(&sealed_product(), config).unwrap();
        let braced = build_sealed_product_interior_mesh(&product, config).expect("braced mesh");

        assert!(
            braced
                .patch(ProductInteriorPatchKind::BraceWall { brace_index: 0 })
                .is_some_and(|patch| patch.area_m2 > 0.0)
        );
        assert!(braced.diagnostics.fluid_volume_m3 < unbraced.diagnostics.fluid_volume_m3);
        assert_ne!(braced.geometry_fingerprint, unbraced.geometry_fingerprint);
        assert!(is_single_fluid_component(&mut braced.mesh.clone()));
    }

    #[test]
    fn brace_lowering_rejects_unresolved_and_overlapping_solids() {
        let mut unresolved = sealed_product();
        unresolved
            .cabinet
            .braces
            .push(crate::cabinet_design::BraceDesign {
                panel: crate::cabinet_design::PanelId::Back,
                split_along: crate::cabinet_design::PanelAxis::Width,
                position_fraction: 0.5,
                cross_section: Some(crate::cabinet_design::BraceCrossSection {
                    width_m: 0.01,
                    depth_m: 0.04,
                }),
            });
        assert!(matches!(
            build_sealed_product_interior_mesh(&unresolved, ProductInteriorMeshConfig::default()),
            Err(ProductInteriorMeshError::BraceUnresolved { index: 0 })
        ));

        let mut overlapping = sealed_product();
        for position_fraction in [0.45, 0.55] {
            overlapping
                .cabinet
                .braces
                .push(crate::cabinet_design::BraceDesign {
                    panel: crate::cabinet_design::PanelId::Back,
                    split_along: crate::cabinet_design::PanelAxis::Width,
                    position_fraction,
                    cross_section: Some(crate::cabinet_design::BraceCrossSection {
                        width_m: 0.08,
                        depth_m: 0.04,
                    }),
                });
        }
        assert!(matches!(
            build_sealed_product_interior_mesh(&overlapping, ProductInteriorMeshConfig::default()),
            Err(ProductInteriorMeshError::BraceSolidIntersection {
                first: 0,
                second: 1
            })
        ));
    }

    #[test]
    fn resolved_port_geometry_uses_shared_product_coordinate_convention() {
        let mut product = sealed_product();
        let mut port = CabinetDesign::default().port.expect("default port");
        port.position_m = [0.04, -0.03];
        port.diameter_m = 0.05;
        port.length_m = 0.12;
        port.wall_thickness_m = Some(0.002);
        product.cabinet.port = Some(port);

        let geometry = ResolvedPortGeometry::from_design(port).expect("resolved port");
        assert_eq!(geometry.mouth_center_m, [0.04, -0.03, 0.0]);
        assert_eq!(geometry.internal_exit_center_m, [0.04, -0.03, -0.12]);
        assert_eq!(geometry.outward_axis, [0.0, 0.0, 1.0]);
        assert!((geometry.bore_radius_m - 0.025).abs() < 1.0e-15);
        assert!((geometry.outer_radius_m - 0.027).abs() < 1.0e-15);
        assert!(
            (geometry.bore_area_m2() - std::f64::consts::PI * 0.025_f64.powi(2)).abs() < 1.0e-15
        );

        let original = geometry_fingerprint(&product, ProductInteriorMeshConfig::default());
        product
            .cabinet
            .port
            .as_mut()
            .expect("port")
            .wall_thickness_m = Some(0.003);
        assert_ne!(
            original,
            geometry_fingerprint(&product, ProductInteriorMeshConfig::default())
        );
    }

    #[test]
    fn ported_mesh_retains_mouth_and_solid_tube_wall_patches() {
        let mut product = sealed_product();
        product.drivers[0].position_m[0] = -0.06;
        let mut port = CabinetDesign::default().port.expect("default port");
        port.position_m = [0.06, 0.0];
        product.cabinet.port = Some(port);
        let mesh =
            build_ported_product_interior_mesh(&product, ProductInteriorMeshConfig::default())
                .expect("ported mesh");
        assert!(
            mesh.patch(ProductInteriorPatchKind::PortMouth)
                .is_some_and(|patch| patch.area_m2 > 0.0)
        );
        assert!(
            mesh.patch(ProductInteriorPatchKind::PortWall)
                .is_some_and(|patch| patch.area_m2 > 0.0)
        );
        assert!(mesh.diagnostics.fluid_volume_m3 < product.cabinet.volume_m3());
    }

    #[test]
    fn port_and_brace_lowerings_compose_with_typed_boundary_patches() {
        let mut product = sealed_product();
        product.drivers[0].position_m[0] = -0.06;
        let mut port = CabinetDesign::default().port.expect("default port");
        port.position_m = [0.06, 0.0];
        product.cabinet.port = Some(port);
        product
            .cabinet
            .braces
            .push(crate::cabinet_design::BraceDesign {
                panel: crate::cabinet_design::PanelId::Top,
                split_along: crate::cabinet_design::PanelAxis::Width,
                position_fraction: 0.5,
                cross_section: Some(crate::cabinet_design::BraceCrossSection {
                    width_m: 0.05,
                    depth_m: 0.05,
                }),
            });
        let mesh = build_ported_product_interior_mesh(
            &product,
            ProductInteriorMeshConfig {
                max_element_size_m: 0.025,
                ..ProductInteriorMeshConfig::default()
            },
        )
        .expect("combined port-and-brace mesh");

        for kind in [
            ProductInteriorPatchKind::PortMouth,
            ProductInteriorPatchKind::PortWall,
            ProductInteriorPatchKind::BraceWall { brace_index: 0 },
        ] {
            assert!(mesh.patch(kind).is_some_and(|patch| patch.area_m2 > 0.0));
        }
        assert!(is_single_fluid_component(&mut mesh.mesh.clone()));
    }

    #[test]
    fn ported_body_fitted_mesh_supports_two_nonintersecting_braces() {
        let mut product = sealed_product();
        product.drivers[0].position_m[0] = -0.06;
        product.drivers[0].radius_m = 0.03;
        let mut port = CabinetDesign::default().port.expect("default port");
        port.position_m = [0.06, 0.0];
        product.cabinet.port = Some(port);
        product.cabinet.braces.extend([
            BraceDesign {
                panel: PanelId::Top,
                split_along: PanelAxis::Width,
                position_fraction: 0.5,
                cross_section: Some(BraceCrossSection {
                    width_m: 0.04,
                    depth_m: 0.04,
                }),
            },
            BraceDesign {
                panel: PanelId::Bottom,
                split_along: PanelAxis::Width,
                position_fraction: 0.5,
                cross_section: Some(BraceCrossSection {
                    width_m: 0.04,
                    depth_m: 0.04,
                }),
            },
        ]);

        let resolved_braces = product
            .cabinet
            .braces
            .iter()
            .copied()
            .enumerate()
            .map(|(index, brace)| {
                ResolvedBraceGeometry::from_design(&product.cabinet, brace, index)
                    .expect("test brace should resolve")
            })
            .collect::<Vec<_>>();
        assert_eq!(
            body_fitted_brace_wall(resolved_braces[0].solid, &product.cabinet),
            Some(InteriorWall::Top)
        );
        assert_eq!(
            body_fitted_brace_wall(resolved_braces[1].solid, &product.cabinet),
            Some(InteriorWall::Bottom)
        );
        let (back_boundary, _) =
            body_fitted_panel_boundary(&product.cabinet, InteriorWall::Back, 24);
        let back_outer = [
            back_boundary[0],
            back_boundary[6],
            back_boundary[12],
            back_boundary[18],
        ]
        .into_iter()
        .map(|[u, v]| (u, v))
        .collect::<Vec<_>>();
        let back_footprints = resolved_braces
            .iter()
            .map(|brace| {
                body_fitted_brace_footprint(brace.solid, &product.cabinet, InteriorWall::Back)
            })
            .collect::<Vec<_>>();
        assert!(triangulate_body_fitted_panel_regions(&back_outer, &back_footprints, &[]).is_ok());
        for wall in InteriorWall::ALL {
            let (boundary, _) = body_fitted_panel_boundary(&product.cabinet, wall, 24);
            let outer = [boundary[0], boundary[6], boundary[12], boundary[18]]
                .into_iter()
                .map(|[u, v]| (u, v))
                .collect::<Vec<_>>();
            let mut holes = product
                .drivers
                .iter()
                .filter(|driver| driver_wall(driver) == Some(wall))
                .map(|driver| {
                    body_fitted_driver_ring(&product.cabinet, wall, driver, boundary[0], 24)
                })
                .collect::<Vec<_>>();
            if wall == InteriorWall::Front {
                let resolved_port =
                    ResolvedPortGeometry::from_design(product.cabinet.port.expect("test port"))
                        .expect("test port should resolve");
                holes.push(body_fitted_circle_ring(
                    [
                        resolved_port.mouth_center_m[0],
                        resolved_port.mouth_center_m[1],
                    ],
                    resolved_port.outer_radius_m,
                    boundary[0],
                    24,
                ));
            }
            let footprints = resolved_braces
                .iter()
                .filter(|brace| body_fitted_brace_touches_wall(brace.solid, &product.cabinet, wall))
                .map(|brace| body_fitted_brace_footprint(brace.solid, &product.cabinet, wall))
                .collect::<Vec<_>>();
            if !triangulate_body_fitted_panel_regions(&outer, &footprints, &holes).is_ok() {
                panic!("PANEL_HOLES_ERROR_{wall:?}");
            }
        }

        let mesh = build_body_fitted_sealed_product_interior_mesh(
            &product,
            ProductInteriorMeshConfig {
                max_element_size_m: 0.025,
                ..ProductInteriorMeshConfig::default()
            },
        )
        .expect("two-brace front-ported body-fitted mesh");

        for kind in [
            ProductInteriorPatchKind::PortMouth,
            ProductInteriorPatchKind::PortWall,
            ProductInteriorPatchKind::BraceWall { brace_index: 0 },
            ProductInteriorPatchKind::BraceWall { brace_index: 1 },
        ] {
            assert!(
                mesh.patch(kind)
                    .is_some_and(|patch| patch.area_m2 > 0.0 && !patch.boundary_faces.is_empty()),
                "missing or empty typed patch {kind:?}"
            );
        }
        assert!(is_single_fluid_component(&mut mesh.mesh.clone()));
        assert!(mesh.diagnostics.fluid_volume_m3 < product.cabinet.volume_m3());
    }

    #[test]
    fn body_fitted_mesh_supports_full_depth_top_and_bottom_braces() {
        let mut product = sealed_product();
        product.drivers[0].position_m[0] = -0.06;
        product.drivers[0].radius_m = 0.03;

        let brace_depth_m = product.cabinet.box_depth_m;
        let brace = |panel| BraceDesign {
            panel,
            split_along: PanelAxis::Height,
            position_fraction: 0.5,
            cross_section: Some(BraceCrossSection {
                width_m: brace_depth_m,
                depth_m: 0.04,
            }),
        };
        product
            .cabinet
            .braces
            .extend([brace(PanelId::Top), brace(PanelId::Bottom)]);

        let mesh = build_body_fitted_sealed_product_interior_mesh(
            &product,
            ProductInteriorMeshConfig {
                max_element_size_m: 0.025,
                ..ProductInteriorMeshConfig::default()
            },
        )
        .expect("full-depth boundary-spanning braces should form a valid PLC");

        for kind in [
            ProductInteriorPatchKind::DriverRear { driver_index: 0 },
            ProductInteriorPatchKind::BraceWall { brace_index: 0 },
            ProductInteriorPatchKind::BraceWall { brace_index: 1 },
        ] {
            let patch = mesh.patch(kind).expect("typed brace/driver patch");
            assert!(patch.area_m2 > 0.0, "{kind:?} must have positive area");
            assert!(!patch.boundary_faces.is_empty(), "{kind:?} must be meshed");
        }
        assert!(is_single_fluid_component(&mut mesh.mesh.clone()));
        assert!(mesh.diagnostics.fluid_volume_m3 < product.cabinet.volume_m3());
        assert!(
            mesh.patch(ProductInteriorPatchKind::RigidWall(InteriorWall::Top))
                .is_none()
        );
        assert!(
            mesh.patch(ProductInteriorPatchKind::RigidWall(InteriorWall::Bottom))
                .is_none()
        );
        assert_eq!(
            mesh.mesh.boundaries.len(),
            mesh.patches
                .iter()
                .map(|patch| patch.boundary_faces.len())
                .sum::<usize>()
        );
    }

    #[test]
    fn port_and_brace_solid_intersection_fails_closed() {
        let mut product = sealed_product();
        product.drivers[0].position_m[0] = -0.08;
        product.drivers[0].radius_m = 0.03;
        let mut port = CabinetDesign::default().port.expect("default port");
        port.position_m = [0.0, 0.0];
        port.length_m = product.cabinet.box_depth_m;
        product.cabinet.port = Some(port);
        product
            .cabinet
            .braces
            .push(crate::cabinet_design::BraceDesign {
                panel: crate::cabinet_design::PanelId::Top,
                split_along: crate::cabinet_design::PanelAxis::Width,
                position_fraction: 0.5,
                cross_section: Some(crate::cabinet_design::BraceCrossSection {
                    width_m: 0.05,
                    depth_m: 0.05,
                }),
            });
        // The top brace does not touch the central port, so use a back-panel
        // rib at the enclosure-side tube exit to exercise the solid overlap gate.
        product.cabinet.braces[0].panel = crate::cabinet_design::PanelId::Back;
        product.cabinet.braces[0].split_along = crate::cabinet_design::PanelAxis::Width;
        let result = build_ported_product_interior_mesh(
            &product,
            ProductInteriorMeshConfig {
                max_element_size_m: 0.025,
                ..ProductInteriorMeshConfig::default()
            },
        );
        assert!(
            matches!(
                result,
                Err(ProductInteriorMeshError::PortBraceIntersection { brace_index: 0 })
            ),
            "unexpected combined-solid admission result: {result:?}"
        );
    }
}
