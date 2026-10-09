//! Product-level loudspeaker design contracts.
//!
//! This module is deliberately independent of the BEM application crate.  It
//! owns the geometry and complex-drive contract that every fidelity level can
//! consume: lumped, resolved FEM, exterior BEM, and the staged optimizer.

use num_complex::Complex64;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;

use super::acoustic_rom::AcousticRomConfig;
use super::calibration::IndependentMeasurementMetadata;
use super::conventions::ArtifactProvenance;
use super::fem_bem_coupling::coupled::{CouplingDrive, DriverVelocityProfile, FemBemCoupling};
use super::fem_bem_coupling::fixtures::CylinderFixtureDiscretization;
use super::fem_bem_coupling::rom::{RomPromotionPolicy, RomPromotionReport};
use super::fem_bem_coupling::{MonolithicSealedProductCoupling, ProductCouplingMesh};
use super::port_flow::{
    PortFlowPromotionEvidence, PortFlowPromotionGate, PortFlowRuntimePromotionArtifact,
    replay_port_flow_promotion, validate_port_flow_promotion,
};
use super::product_acoustic_mesh::{
    ProductInteriorMeshConfig, ProductInteriorPatchKind,
    build_body_fitted_sealed_product_interior_mesh,
};
use super::structural_acoustic::{StructuralAcousticFeedback, StructuralAcousticResponse};
use super::structural_constraints::{CabinetPanelEvaluation, cabinet_panel_response};
use super::thermal_network::{
    DriverThermalNetworkConfig, THERMAL_NODE_COUNT, ThermalNetworkPower, ThermalNetworkState,
    ThermalNode,
};
use super::waveguide_design::ProductWaveguideDesign;
use super::{AcousticMedium, CabinetDesign, CabinetDesignError, EnclosureSpec};
use sonium_solvers::LinearSolveConfig;

/// The idealized passive branch shape used by a product crossover.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CrossoverKind {
    /// No frequency shaping.
    FullRange,
    /// First-order low-pass branch.
    LowPass,
    /// First-order high-pass branch.
    HighPass,
    /// Second-order Linkwitz-Riley low-pass.
    LinkwitzRiley2LowPass,
    /// Second-order Linkwitz-Riley high-pass with polarity reversal for summation.
    LinkwitzRiley2HighPass,
    /// Fourth-order Linkwitz-Riley low-pass.
    LinkwitzRiley4LowPass,
    /// Fourth-order Linkwitz-Riley high-pass.
    LinkwitzRiley4HighPass,
}

/// One complex crossover branch.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct CrossoverBranch {
    pub kind: CrossoverKind,
    /// Corner frequency for filtered branches; ignored by `FullRange`.
    pub corner_frequency_hz: f64,
    /// Real non-negative branch gain.
    pub gain: f64,
    /// Propagation/electrical delay applied after the branch response.
    pub delay_s: f64,
}

impl Default for CrossoverBranch {
    fn default() -> Self {
        Self {
            kind: CrossoverKind::FullRange,
            corner_frequency_hz: 1_000.0,
            gain: 1.0,
            delay_s: 0.0,
        }
    }
}

impl CrossoverBranch {
    pub fn validate(self) -> Result<(), ProductDesignError> {
        if !self.corner_frequency_hz.is_finite() || self.corner_frequency_hz <= 0.0 {
            return Err(ProductDesignError::InvalidCrossover(
                "corner frequency must be finite and positive",
            ));
        }
        if !self.gain.is_finite() || self.gain < 0.0 {
            return Err(ProductDesignError::InvalidCrossover(
                "gain must be finite and non-negative",
            ));
        }
        if !self.delay_s.is_finite() || self.delay_s < 0.0 {
            return Err(ProductDesignError::InvalidCrossover(
                "delay must be finite and non-negative",
            ));
        }
        Ok(())
    }

    /// Evaluate the branch as a complex RMS voltage transfer.
    pub fn transfer_at(self, frequency_hz: f64) -> Result<Complex64, ProductDesignError> {
        self.validate()?;
        if !frequency_hz.is_finite() || frequency_hz <= 0.0 {
            return Err(ProductDesignError::InvalidFrequency);
        }
        let omega = std::f64::consts::TAU * frequency_hz;
        let ratio = frequency_hz / self.corner_frequency_hz;
        let jw = Complex64::new(0.0, ratio);
        let shape = match self.kind {
            CrossoverKind::FullRange => Complex64::new(1.0, 0.0),
            CrossoverKind::LowPass => Complex64::new(1.0, 0.0) / (Complex64::new(1.0, 0.0) + jw),
            CrossoverKind::HighPass => jw / (Complex64::new(1.0, 0.0) + jw),
            CrossoverKind::LinkwitzRiley2LowPass => (1.0 / (1.0 + jw)).powu(2),
            CrossoverKind::LinkwitzRiley2HighPass => -(jw / (1.0 + jw)).powu(2),
            CrossoverKind::LinkwitzRiley4LowPass => {
                (1.0 / (1.0 + std::f64::consts::SQRT_2 * jw + jw * jw)).powu(2)
            }
            CrossoverKind::LinkwitzRiley4HighPass => {
                (jw * jw / (1.0 + std::f64::consts::SQRT_2 * jw + jw * jw)).powu(2)
            }
        };
        let delay = Complex64::from_polar(1.0, -omega * self.delay_s);
        Ok(shape * self.gain * delay)
    }
}

/// Collection of independent crossover branches, one per driver.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CrossoverNetwork {
    pub branches: Vec<CrossoverBranch>,
}

impl CrossoverNetwork {
    pub fn validate(&self, driver_count: usize) -> Result<(), ProductDesignError> {
        if self.branches.len() != driver_count {
            return Err(ProductDesignError::CrossoverCountMismatch {
                expected: driver_count,
                got: self.branches.len(),
            });
        }
        for branch in &self.branches {
            branch.validate()?;
        }
        Ok(())
    }

    pub fn transfers_at(
        &self,
        driver_count: usize,
        frequency_hz: f64,
    ) -> Result<Vec<Complex64>, ProductDesignError> {
        self.validate(driver_count)?;
        self.branches
            .iter()
            .copied()
            .map(|branch| branch.transfer_at(frequency_hz))
            .collect()
    }
}

/// A physically placed driver on a rectangular product-cabinet face.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProductDriver {
    pub id: String,
    pub radius_m: f64,
    /// `[x, y, z]` position in the product frame: the front face is `z = 0`,
    /// the back face is `z = -box_depth_m`.
    pub position_m: [f64; 3],
    /// Outward unit normal of the diaphragm in the product frame.
    ///
    /// The resolution-limited Cartesian interior mesh supports the six
    /// axis-aligned rectangular cabinet faces. Tilted apertures still require
    /// body-fitted surface meshing and are rejected rather than silently
    /// meshing the wrong geometry.
    #[serde(default = "default_driver_normal")]
    pub normal: [f64; 3],
    pub force_factor_tm: f64,
    pub coil_resistance_ohm: f64,
    pub coil_reactance_ohm: f64,
    pub mechanical_resistance_n_s_per_m: f64,
    pub mechanical_reactance_n_s_per_m: f64,
    pub crossover_branch: usize,
}

const fn default_driver_normal() -> [f64; 3] {
    [0.0, 0.0, 1.0]
}

/// Cabinet face on which a driver aperture is mounted.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DriverMountingFace {
    Front,
    Back,
    Right,
    Left,
    Top,
    Bottom,
}

impl DriverMountingFace {
    /// Outward unit normal of this cabinet face.
    pub const fn outward_normal(self) -> [f64; 3] {
        match self {
            Self::Front => [0.0, 0.0, 1.0],
            Self::Back => [0.0, 0.0, -1.0],
            Self::Right => [1.0, 0.0, 0.0],
            Self::Left => [-1.0, 0.0, 0.0],
            Self::Top => [0.0, 1.0, 0.0],
            Self::Bottom => [0.0, -1.0, 0.0],
        }
    }
}

/// Geometry frame shared by oriented body-fitted driver lowerings.
///
/// `wall_center_m` is the declared product position on the cabinet face.
/// `center_m` is the physical aperture-plane center shifted inward just far
/// enough that the tilted disk remains inside the cabinet. The projected ring
/// is the elliptical footprint where the aperture throat meets the cabinet
/// face; this distinction prevents a tilted disk from being silently treated
/// as a circular planar wall hole.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct DriverApertureFrame {
    pub mounting_face: DriverMountingFace,
    pub wall_center_m: [f64; 3],
    pub center_m: [f64; 3],
    pub wall_normal_m: [f64; 3],
    pub normal_m: [f64; 3],
    pub tangent_u_m: [f64; 3],
    pub tangent_v_m: [f64; 3],
    pub radius_m: f64,
    pub projected_radius_m: f64,
    pub inward_offset_m: f64,
}

impl DriverApertureFrame {
    /// Point on the physical circular aperture plane at a polar angle.
    pub fn aperture_point(self, angle_rad: f64) -> [f64; 3] {
        let radial = [
            self.tangent_u_m[0] * angle_rad.cos() + self.tangent_v_m[0] * angle_rad.sin(),
            self.tangent_u_m[1] * angle_rad.cos() + self.tangent_v_m[1] * angle_rad.sin(),
            self.tangent_u_m[2] * angle_rad.cos() + self.tangent_v_m[2] * angle_rad.sin(),
        ];
        [
            self.center_m[0] + self.radius_m * radial[0],
            self.center_m[1] + self.radius_m * radial[1],
            self.center_m[2] + self.radius_m * radial[2],
        ]
    }

    /// Project an aperture-plane point orthogonally onto its cabinet face.
    pub fn project_to_wall(self, point: [f64; 3]) -> [f64; 3] {
        let offset = [
            point[0] - self.wall_center_m[0],
            point[1] - self.wall_center_m[1],
            point[2] - self.wall_center_m[2],
        ];
        let distance = offset[0] * self.wall_normal_m[0]
            + offset[1] * self.wall_normal_m[1]
            + offset[2] * self.wall_normal_m[2];
        [
            point[0] - distance * self.wall_normal_m[0],
            point[1] - distance * self.wall_normal_m[1],
            point[2] - distance * self.wall_normal_m[2],
        ]
    }
}

fn dot3(left: [f64; 3], right: [f64; 3]) -> f64 {
    left[0] * right[0] + left[1] * right[1] + left[2] * right[2]
}

fn cross3(left: [f64; 3], right: [f64; 3]) -> [f64; 3] {
    [
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    ]
}

fn norm3(value: [f64; 3]) -> f64 {
    dot3(value, value).sqrt()
}

fn normalized3(value: [f64; 3]) -> Option<[f64; 3]> {
    let norm = norm3(value);
    (norm.is_finite() && norm > f64::EPSILON)
        .then(|| [value[0] / norm, value[1] / norm, value[2] / norm])
}

impl ProductDriver {
    pub fn validate(
        &self,
        index: usize,
        baffle_width_m: f64,
        baffle_height_m: f64,
        box_width_m: f64,
        box_height_m: f64,
        box_depth_m: f64,
    ) -> Result<(), ProductDesignError> {
        if self.id.trim().is_empty() {
            return Err(ProductDesignError::InvalidDriver {
                index,
                reason: "driver id must not be blank",
            });
        }
        for (name, value) in [
            ("radius", self.radius_m),
            ("force factor", self.force_factor_tm),
        ] {
            if !value.is_finite() || value <= 0.0 {
                return Err(ProductDesignError::InvalidDriver {
                    index,
                    reason: name,
                });
            }
        }
        for value in self.position_m {
            if !value.is_finite() {
                return Err(ProductDesignError::InvalidDriver {
                    index,
                    reason: "position must be finite",
                });
            }
        }
        let normal_norm_sq = self.normal.iter().map(|value| value * value).sum::<f64>();
        if self.normal.iter().any(|value| !value.is_finite())
            || !normal_norm_sq.is_finite()
            || (normal_norm_sq.sqrt() - 1.0).abs() > 1.0e-9
        {
            return Err(ProductDesignError::InvalidDriver {
                index,
                reason: "driver normal must be finite and unit length",
            });
        }
        if self
            .aperture_frame_unchecked(
                baffle_width_m,
                baffle_height_m,
                box_width_m,
                box_height_m,
                box_depth_m,
            )
            .is_err()
        {
            return Err(ProductDesignError::DriverOutsideBaffle { index });
        }
        if !self.coil_resistance_ohm.is_finite()
            || self.coil_resistance_ohm <= 0.0
            || !self.coil_reactance_ohm.is_finite()
            || !self.mechanical_resistance_n_s_per_m.is_finite()
            || self.mechanical_resistance_n_s_per_m < 0.0
            || !self.mechanical_reactance_n_s_per_m.is_finite()
        {
            return Err(ProductDesignError::InvalidDriver {
                index,
                reason: "impedance values must be finite and passive",
            });
        }
        Ok(())
    }

    /// Resolve the physically oriented aperture frame for a validated cabinet
    /// placement. Tilt is limited to 60 degrees from the host cabinet face so
    /// the projected elliptical throat remains bounded and well-conditioned.
    pub fn aperture_frame(
        &self,
        baffle_width_m: f64,
        baffle_height_m: f64,
        box_width_m: f64,
        box_height_m: f64,
        box_depth_m: f64,
    ) -> Result<DriverApertureFrame, ProductDesignError> {
        self.validate(
            0,
            baffle_width_m,
            baffle_height_m,
            box_width_m,
            box_height_m,
            box_depth_m,
        )?;
        self.aperture_frame_unchecked(
            baffle_width_m,
            baffle_height_m,
            box_width_m,
            box_height_m,
            box_depth_m,
        )
        .map_err(|reason| ProductDesignError::InvalidDriver { index: 0, reason })
    }

    /// Whether the aperture plane is coincident with its cabinet wall.
    pub fn is_axis_aligned(&self) -> bool {
        self.normal
            .iter()
            .enumerate()
            .all(|(axis, value)| value.abs() >= 1.0 - 1.0e-9 || self.normal[axis].abs() <= 1.0e-9)
    }

    fn aperture_frame_unchecked(
        &self,
        baffle_width_m: f64,
        baffle_height_m: f64,
        box_width_m: f64,
        box_height_m: f64,
        box_depth_m: f64,
    ) -> Result<DriverApertureFrame, &'static str> {
        let axis = (0..3)
            .max_by(|left, right| {
                self.normal[*left]
                    .abs()
                    .total_cmp(&self.normal[*right].abs())
            })
            .ok_or("driver normal has no dominant mounting face")?;
        let sign = self.normal[axis].signum();
        let mounting_face = match (axis, sign as i8) {
            (0, 1) => DriverMountingFace::Right,
            (0, -1) => DriverMountingFace::Left,
            (1, 1) => DriverMountingFace::Top,
            (1, -1) => DriverMountingFace::Bottom,
            (2, 1) => DriverMountingFace::Front,
            (2, -1) => DriverMountingFace::Back,
            _ => return Err("driver normal has invalid mounting face"),
        };
        let wall_normal_m = mounting_face.outward_normal();
        let face_cosine = dot3(self.normal, wall_normal_m);
        if !face_cosine.is_finite() || face_cosine < 0.5 {
            return Err("driver normal must point outward within 60 degrees of its cabinet face");
        }
        let projected_radius_m = self.radius_m / face_cosine;
        let on_face = match mounting_face {
            DriverMountingFace::Front => {
                self.position_m[2].abs() <= 1.0e-9
                    && self.position_m[0].abs() + projected_radius_m <= baffle_width_m * 0.5
                    && self.position_m[1].abs() + projected_radius_m <= baffle_height_m * 0.5
            }
            DriverMountingFace::Back => {
                (self.position_m[2] + box_depth_m).abs() <= 1.0e-9
                    && self.position_m[0].abs() + projected_radius_m <= baffle_width_m * 0.5
                    && self.position_m[1].abs() + projected_radius_m <= baffle_height_m * 0.5
            }
            DriverMountingFace::Right => {
                (self.position_m[0] - box_width_m * 0.5).abs() <= 1.0e-9
                    && self.position_m[1].abs() + projected_radius_m <= box_height_m * 0.5
                    && self.position_m[2] - projected_radius_m >= -box_depth_m
                    && self.position_m[2] + projected_radius_m <= 0.0
            }
            DriverMountingFace::Left => {
                (self.position_m[0] + box_width_m * 0.5).abs() <= 1.0e-9
                    && self.position_m[1].abs() + projected_radius_m <= box_height_m * 0.5
                    && self.position_m[2] - projected_radius_m >= -box_depth_m
                    && self.position_m[2] + projected_radius_m <= 0.0
            }
            DriverMountingFace::Top => {
                (self.position_m[1] - box_height_m * 0.5).abs() <= 1.0e-9
                    && self.position_m[0].abs() + projected_radius_m <= box_width_m * 0.5
                    && self.position_m[2] - projected_radius_m >= -box_depth_m
                    && self.position_m[2] + projected_radius_m <= 0.0
            }
            DriverMountingFace::Bottom => {
                (self.position_m[1] + box_height_m * 0.5).abs() <= 1.0e-9
                    && self.position_m[0].abs() + projected_radius_m <= box_width_m * 0.5
                    && self.position_m[2] - projected_radius_m >= -box_depth_m
                    && self.position_m[2] + projected_radius_m <= 0.0
            }
        };
        if !on_face {
            return Err("oriented aperture footprint lies outside its cabinet face");
        }

        let reference = if self.normal[0].abs() < 0.8 {
            [1.0, 0.0, 0.0]
        } else {
            [0.0, 1.0, 0.0]
        };
        let tangent_u_m = normalized3(cross3(reference, self.normal))
            .ok_or("driver normal cannot construct a stable aperture basis")?;
        let tangent_v_m = normalized3(cross3(self.normal, tangent_u_m))
            .ok_or("driver normal cannot construct a stable aperture basis")?;
        let tilt = (1.0 - face_cosine * face_cosine).max(0.0).sqrt();
        let inward_offset_m = if tilt <= 1.0e-12 {
            0.0
        } else {
            self.radius_m * tilt + self.radius_m * 1.0e-6
        };
        let center_m = [
            self.position_m[0] - inward_offset_m * wall_normal_m[0],
            self.position_m[1] - inward_offset_m * wall_normal_m[1],
            self.position_m[2] - inward_offset_m * wall_normal_m[2],
        ];
        Ok(DriverApertureFrame {
            mounting_face,
            wall_center_m: self.position_m,
            center_m,
            wall_normal_m,
            normal_m: self.normal,
            tangent_u_m,
            tangent_v_m,
            radius_m: self.radius_m,
            projected_radius_m,
            inward_offset_m,
        })
    }

    pub fn coil_impedance_ohm(&self) -> Complex64 {
        Complex64::new(self.coil_resistance_ohm, self.coil_reactance_ohm)
    }

    pub fn mechanical_impedance_n_s_per_m(&self) -> Complex64 {
        Complex64::new(
            self.mechanical_resistance_n_s_per_m,
            self.mechanical_reactance_n_s_per_m,
        )
    }
}

/// Complete product-level geometry and complex-drive definition.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ProductDesign {
    pub cabinet: CabinetDesign,
    pub drivers: Vec<ProductDriver>,
    pub crossover: CrossoverNetwork,
    /// Optional body-fitted waveguide owned by this product design.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub waveguide: Option<ProductWaveguideDesign>,
    /// Optional frequency-domain panel/cone structural feedback. When set,
    /// the speaker BEM feeds its solved cabinet-wall pressure into this modal
    /// model and feeds the resulting normal velocity back into the boundary.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub structural_acoustic: Option<StructuralAcousticFeedback>,
}

/// Wire representation for untrusted product-design artifacts.
///
/// Product fields remain public for in-process optimization, but serialized
/// input must cross the same invariant gate as explicit product construction.
#[derive(Debug, Deserialize)]
struct ProductDesignWire {
    cabinet: CabinetDesign,
    drivers: Vec<ProductDriver>,
    crossover: CrossoverNetwork,
    #[serde(default)]
    waveguide: Option<ProductWaveguideDesign>,
    #[serde(default)]
    structural_acoustic: Option<StructuralAcousticFeedback>,
}

impl<'de> Deserialize<'de> for ProductDesign {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let wire = ProductDesignWire::deserialize(deserializer)?;
        let design = Self {
            cabinet: wire.cabinet,
            drivers: wire.drivers,
            crossover: wire.crossover,
            waveguide: wire.waveguide,
            structural_acoustic: wire.structural_acoustic,
        };
        design
            .validate()
            .map_err(|error| <D::Error as serde::de::Error>::custom(error.to_string()))?;
        Ok(design)
    }
}

/// Versioned, provenance-carrying product input artifact.
///
/// `ProductDesign` remains the in-process optimization type. This wrapper is
/// the serialized ownership boundary for consumers that need to retain the
/// exact product geometry, material mapping, driver identities, crossover,
/// waveguide, and structural-feedback configuration together.
pub const PRODUCT_DESIGN_ARTIFACT_SCHEMA_VERSION: u32 = 2;

#[derive(Debug, Serialize)]
struct ProductDesignArtifactPayload<'a> {
    schema_version: u32,
    design: &'a ProductDesign,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProductDesignArtifact {
    pub schema_version: u32,
    pub provenance: ArtifactProvenance,
    pub design: ProductDesign,
    /// Lowercase SHA-256 of the canonical schema-versioned product payload.
    /// Provenance is intentionally excluded so a deterministic preview can be
    /// reclassified without changing the product-content identity.
    pub content_sha256: String,
}

impl ProductDesignArtifact {
    /// Create a native deterministic artifact from a validated product.
    pub fn from_design(design: &ProductDesign) -> Result<Self, ProductDesignError> {
        design.validate()?;
        let mut artifact = Self {
            schema_version: PRODUCT_DESIGN_ARTIFACT_SCHEMA_VERSION,
            provenance: ArtifactProvenance::native_golden(),
            design: design.clone(),
            content_sha256: String::new(),
        };
        artifact.content_sha256 = artifact
            .payload_sha256()
            .map_err(|error| ProductDesignError::InvalidProductArtifact(error.to_string()))?;
        artifact.validate()?;
        Ok(artifact)
    }

    /// Override provenance for a backend-specific preview or replay envelope.
    pub fn with_provenance(mut self, provenance: ArtifactProvenance) -> Self {
        self.provenance = provenance;
        self
    }

    /// Validate the artifact envelope and its complete product payload.
    pub fn validate(&self) -> Result<(), ProductDesignError> {
        if self.schema_version != PRODUCT_DESIGN_ARTIFACT_SCHEMA_VERSION {
            return Err(ProductDesignError::InvalidProductArtifact(format!(
                "unsupported product artifact schema {}; expected {}",
                self.schema_version, PRODUCT_DESIGN_ARTIFACT_SCHEMA_VERSION
            )));
        }
        if self.provenance.schema_version != super::conventions::ARTIFACT_PROVENANCE_SCHEMA_VERSION
        {
            return Err(ProductDesignError::InvalidProductArtifact(
                "unsupported product artifact provenance schema".into(),
            ));
        }
        if !self.provenance.deterministic {
            return Err(ProductDesignError::InvalidProductArtifact(
                "product input artifact provenance must be deterministic".into(),
            ));
        }
        if self.content_sha256.len() != 64
            || !self
                .content_sha256
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        {
            return Err(ProductDesignError::InvalidProductArtifact(
                "product artifact content digest must be lowercase SHA-256".into(),
            ));
        }
        let expected_digest = self
            .payload_sha256()
            .map_err(|error| ProductDesignError::InvalidProductArtifact(error.to_string()))?;
        if self.content_sha256 != expected_digest {
            return Err(ProductDesignError::InvalidProductArtifact(
                "product artifact content digest mismatch".into(),
            ));
        }
        self.design.validate()
    }

    /// Serialize stable field ordering and bounded floating-point precision.
    pub fn canonical_json(&self) -> Result<Vec<u8>, serde_json::Error> {
        let mut value = serde_json::to_value(self)?;
        canonicalize_product_json(&mut value);
        serde_json::to_vec(&value)
    }

    fn payload_sha256(&self) -> Result<String, serde_json::Error> {
        let mut value = serde_json::to_value(ProductDesignArtifactPayload {
            schema_version: self.schema_version,
            design: &self.design,
        })?;
        canonicalize_product_json(&mut value);
        let bytes = serde_json::to_vec(&value)?;
        let digest = Sha256::digest(bytes);
        Ok(digest.iter().map(|byte| format!("{byte:02x}")).collect())
    }

    /// Deserialize and validate an artifact before exposing its product.
    pub fn from_json(bytes: &[u8]) -> Result<Self, ProductDesignError> {
        let artifact: Self = serde_json::from_slice(bytes)
            .map_err(|error| ProductDesignError::InvalidProductArtifact(error.to_string()))?;
        artifact.validate()?;
        Ok(artifact)
    }

    /// Consume a validated artifact and return its product input.
    pub fn into_design(self) -> Result<ProductDesign, ProductDesignError> {
        self.validate()?;
        Ok(self.design)
    }
}

impl ProductDesign {
    /// Export this product input through the versioned artifact boundary.
    pub fn to_artifact(&self) -> Result<ProductDesignArtifact, ProductDesignError> {
        ProductDesignArtifact::from_design(self)
    }
}

fn canonicalize_product_json(value: &mut serde_json::Value) {
    match value {
        serde_json::Value::Array(values) => {
            for value in values {
                canonicalize_product_json(value);
            }
        }
        serde_json::Value::Object(values) => {
            let mut entries = std::mem::take(values).into_iter().collect::<Vec<_>>();
            entries.sort_by(|left, right| left.0.cmp(&right.0));
            let mut ordered = serde_json::Map::new();
            for (key, mut value) in entries {
                canonicalize_product_json(&mut value);
                ordered.insert(key, value);
            }
            *values = ordered;
        }
        serde_json::Value::Number(number) => {
            if number.as_i64().is_some() || number.as_u64().is_some() {
                return;
            }
            if let Some(value) = number.as_f64()
                && let Some(canonical) =
                    serde_json::Number::from_f64(canonicalize_product_float(value))
            {
                *number = canonical;
            }
        }
        serde_json::Value::Null | serde_json::Value::Bool(_) | serde_json::Value::String(_) => {}
    }
}

fn canonicalize_product_float(value: f64) -> f64 {
    if value == 0.0 || !value.is_finite() {
        return value;
    }
    let exponent = value.abs().log10().floor() as i32;
    let scale = 10_f64.powi(12 - exponent - 1);
    (value * scale).round() / scale
}

/// Current serialized schema version for product thermal evidence.
pub const PRODUCT_THERMAL_EVIDENCE_SCHEMA_VERSION: u32 = 1;

/// Thermal-network evidence retained beside a product reference point.
///
/// This is deliberately separate from the acoustic state: thermal sources are
/// supplied by the electromechanical solve and the state is produced by the
/// validated conjugate thermal network. Product promotion can therefore
/// reject a reference that has acoustic closure but no thermal closure.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProductThermalReferenceEvidence {
    pub schema_version: u32,
    pub driver_ids: Vec<String>,
    pub source_power_w: Vec<ThermalNetworkPower>,
    pub states: Vec<ThermalNetworkState>,
    /// Optional hot copper resistances from a converged electro-thermal
    /// fixed point. `None` preserves compatibility with legacy one-way
    /// thermal evidence.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hot_coil_resistance_ohm: Option<Vec<f64>>,
    /// Number of resistance/temperature iterations used by feedback.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub feedback_iterations: Option<usize>,
    /// Final maximum relative resistance update at the feedback fixed point.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub feedback_relative_resistance_change: Option<f64>,
}

/// Acceptance limits for product thermal evidence.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ProductThermalPromotionGate {
    pub max_balance_residual_w: f64,
    pub max_temperature_k: f64,
    /// Require converged hot-copper feedback metadata.
    /// Legacy one-way thermal evidence keeps this false by default.
    #[serde(default)]
    pub require_feedback_closure: bool,
}

impl Default for ProductThermalPromotionGate {
    fn default() -> Self {
        Self {
            max_balance_residual_w: 1.0e-8,
            max_temperature_k: 450.0,
            require_feedback_closure: false,
        }
    }
}

impl ProductThermalPromotionGate {
    fn validate(self) -> Result<(), ProductDesignError> {
        if !self.max_balance_residual_w.is_finite()
            || self.max_balance_residual_w < 0.0
            || !self.max_temperature_k.is_finite()
            || self.max_temperature_k <= 0.0
        {
            return Err(ProductDesignError::InvalidReferenceEvidence(
                "invalid product thermal promotion gate".into(),
            ));
        }
        Ok(())
    }
}

impl ProductThermalReferenceEvidence {
    fn validate_for_driver_count(
        &self,
        expected_driver_count: usize,
        gate: ProductThermalPromotionGate,
    ) -> Result<(), ProductDesignError> {
        gate.validate()?;
        if gate.require_feedback_closure
            && (self.hot_coil_resistance_ohm.is_none()
                || self.feedback_iterations.is_none()
                || self.feedback_relative_resistance_change.is_none())
        {
            return Err(ProductDesignError::InvalidReferenceEvidence(
                "product thermal evidence is missing electro-thermal feedback closure".into(),
            ));
        }
        if self.schema_version != PRODUCT_THERMAL_EVIDENCE_SCHEMA_VERSION
            || self.driver_ids.len() != expected_driver_count
            || self.source_power_w.len() != expected_driver_count
            || self.states.len() != expected_driver_count
            || expected_driver_count == 0
        {
            return Err(ProductDesignError::InvalidReferenceEvidence(
                "invalid product thermal evidence shape or schema".into(),
            ));
        }
        let mut ids = BTreeSet::new();
        for id in &self.driver_ids {
            if id.trim().is_empty() || !ids.insert(id.as_str()) {
                return Err(ProductDesignError::InvalidReferenceEvidence(
                    "product thermal evidence has empty or duplicate driver identity".into(),
                ));
            }
        }
        for (power, state) in self.source_power_w.iter().zip(&self.states) {
            power.validate().map_err(|error| {
                ProductDesignError::InvalidReferenceEvidence(format!(
                    "invalid product thermal source power: {error}"
                ))
            })?;
            if state.power_w != *power
                || state.maximum_balance_residual_w < 0.0
                || !state.maximum_balance_residual_w.is_finite()
                || state
                    .temperatures_k
                    .iter()
                    .any(|temperature| !temperature.is_finite() || *temperature <= 0.0)
                || state.temperatures_k.len() != THERMAL_NODE_COUNT
            {
                return Err(ProductDesignError::InvalidReferenceEvidence(
                    "invalid product thermal state".into(),
                ));
            }
            if state.maximum_balance_residual_w > gate.max_balance_residual_w
                || state.temperatures_k.iter().copied().fold(0.0, f64::max) > gate.max_temperature_k
            {
                return Err(ProductDesignError::InvalidReferenceEvidence(
                    "product thermal evidence exceeds promotion gate".into(),
                ));
            }
        }
        if let Some(resistances) = &self.hot_coil_resistance_ohm {
            if resistances.len() != self.driver_ids.len()
                || resistances
                    .iter()
                    .any(|resistance| !resistance.is_finite() || *resistance <= 0.0)
            {
                return Err(ProductDesignError::InvalidReferenceEvidence(
                    "invalid product thermal hot copper resistances".into(),
                ));
            }
            if self
                .feedback_iterations
                .is_some_and(|iterations| iterations == 0)
            {
                return Err(ProductDesignError::InvalidReferenceEvidence(
                    "product thermal feedback iterations must be positive".into(),
                ));
            }
            if self
                .feedback_relative_resistance_change
                .is_some_and(|change| !change.is_finite() || change < 0.0)
            {
                return Err(ProductDesignError::InvalidReferenceEvidence(
                    "invalid product thermal feedback residual".into(),
                ));
            }
        } else if self.feedback_iterations.is_some()
            || self.feedback_relative_resistance_change.is_some()
        {
            return Err(ProductDesignError::InvalidReferenceEvidence(
                "product thermal feedback metadata requires hot copper resistances".into(),
            ));
        }
        Ok(())
    }

    fn validate_for_driver_ids(
        &self,
        expected_driver_ids: &[String],
        gate: ProductThermalPromotionGate,
    ) -> Result<(), ProductDesignError> {
        self.validate_for_driver_count(expected_driver_ids.len(), gate)?;
        if self.driver_ids != expected_driver_ids {
            return Err(ProductDesignError::InvalidReferenceEvidence(
                "product thermal evidence driver identities differ from design".into(),
            ));
        }
        Ok(())
    }
}

/// One validated point from the product-owned distributed FEM–BEM reference.
///
/// This is deliberately a reduced evidence record rather than a serialized
/// field solution. The full nodal state remains owned by the native reference
/// solver; candidate evaluation can consume these observables together with
/// the residual gate without accidentally treating a reduced artifact as a
/// golden field solution.
/// Current serialized schema version for product structural evidence.
pub const PRODUCT_STRUCTURAL_EVIDENCE_SCHEMA_VERSION: u32 = 1;

/// Structural response and convergence evidence retained beside a product
/// reference point. A product reference may only claim structural feedback
/// promotion when the response is tied to the same pressure/frequency point,
/// has finite passive observables, and records iteration and mesh diagnostics.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ProductStructuralReferenceEvidence {
    pub schema_version: u32,
    pub response: StructuralAcousticResponse,
    pub iterations: usize,
    pub relative_velocity_residual: f64,
    /// Optional residual of the explicit structural modal rows in the
    /// product FEM–BEM solve. Legacy reduced-response evidence omits it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub explicit_modal_relative_residual: Option<f64>,
    pub mesh_convergence_error: f64,
}

/// Acceptance limits for product structural feedback evidence.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ProductStructuralPromotionGate {
    pub max_relative_velocity_residual: f64,
    pub max_mesh_convergence_error: f64,
    /// Require residual evidence from the explicit structural modal rows.
    /// Legacy reduced-response evidence keeps this false by default.
    #[serde(default)]
    pub require_explicit_modal_residual: bool,
}

impl Default for ProductStructuralPromotionGate {
    fn default() -> Self {
        Self {
            max_relative_velocity_residual: 1.0e-8,
            max_mesh_convergence_error: 1.0e-2,
            require_explicit_modal_residual: false,
        }
    }
}

impl ProductStructuralPromotionGate {
    fn validate(self) -> Result<(), ProductDesignError> {
        if !self.max_relative_velocity_residual.is_finite()
            || self.max_relative_velocity_residual < 0.0
            || !self.max_mesh_convergence_error.is_finite()
            || self.max_mesh_convergence_error < 0.0
        {
            return Err(ProductDesignError::InvalidReferenceEvidence(
                "product structural promotion gate must be finite and non-negative".into(),
            ));
        }
        Ok(())
    }
}

impl ProductStructuralReferenceEvidence {
    /// Build and validate evidence from a retained structural response.
    pub fn from_response(
        response: StructuralAcousticResponse,
        iterations: usize,
        relative_velocity_residual: f64,
        mesh_convergence_error: f64,
    ) -> Result<Self, ProductDesignError> {
        let evidence = Self {
            schema_version: PRODUCT_STRUCTURAL_EVIDENCE_SCHEMA_VERSION,
            response,
            iterations,
            relative_velocity_residual,
            explicit_modal_relative_residual: None,
            mesh_convergence_error,
        };
        evidence.validate_for_point(
            response.frequency_hz,
            response.pressure_rms_pa,
            ProductStructuralPromotionGate::default(),
        )?;
        Ok(evidence)
    }

    /// Evaluate the production reduced structural-feedback model and retain
    /// its response as promotion evidence.
    pub fn from_feedback(
        feedback: &StructuralAcousticFeedback,
        frequency_hz: f64,
        pressure_rms_pa: Complex64,
        iterations: usize,
        relative_velocity_residual: f64,
        mesh_convergence_error: f64,
    ) -> Result<Self, ProductDesignError> {
        let response = feedback
            .response_at(frequency_hz, pressure_rms_pa)
            .map_err(|error| ProductDesignError::InvalidStructuralFeedback(error.to_string()))?;
        Self::from_response(
            response,
            iterations,
            relative_velocity_residual,
            mesh_convergence_error,
        )
    }

    fn validate_for_point(
        &self,
        frequency_hz: f64,
        pressure_rms_pa: Complex64,
        gate: ProductStructuralPromotionGate,
    ) -> Result<(), ProductDesignError> {
        gate.validate()?;
        if self.schema_version != PRODUCT_STRUCTURAL_EVIDENCE_SCHEMA_VERSION {
            return Err(ProductDesignError::InvalidReferenceEvidence(
                "product structural evidence schema version is unsupported".into(),
            ));
        }
        if self.iterations == 0
            || !self.response.frequency_hz.is_finite()
            || self.response.frequency_hz <= 0.0
            || !self.relative_velocity_residual.is_finite()
            || self.relative_velocity_residual < 0.0
            || !self.mesh_convergence_error.is_finite()
            || self.mesh_convergence_error < 0.0
        {
            return Err(ProductDesignError::InvalidReferenceEvidence(
                "product structural evidence has invalid convergence metadata".into(),
            ));
        }
        if !frequency_hz.is_finite()
            || frequency_hz <= 0.0
            || relative_scalar_error(self.response.frequency_hz, frequency_hz, 1.0) > 1.0e-12
            || relative_complex_error(self.response.pressure_rms_pa, pressure_rms_pa, 1.0e-12)
                > 1.0e-10
        {
            return Err(ProductDesignError::InvalidReferenceEvidence(
                "product structural evidence does not match reference excitation".into(),
            ));
        }
        let response_values = [
            self.response.pressure_rms_pa.re,
            self.response.pressure_rms_pa.im,
            self.response.displacement_rms_m.re,
            self.response.displacement_rms_m.im,
            self.response.velocity_rms_m_per_s.re,
            self.response.velocity_rms_m_per_s.im,
            self.response.stress_rms_pa.re,
            self.response.stress_rms_pa.im,
            self.response.volume_velocity_rms_m3_per_s.re,
            self.response.volume_velocity_rms_m3_per_s.im,
            self.response.absorbed_power_w,
        ];
        if response_values.iter().any(|value| !value.is_finite())
            || self.response.absorbed_power_w < 0.0
        {
            return Err(ProductDesignError::InvalidReferenceEvidence(
                "product structural response must be finite and passive".into(),
            ));
        }
        if self.relative_velocity_residual > gate.max_relative_velocity_residual
            || self.mesh_convergence_error > gate.max_mesh_convergence_error
        {
            return Err(ProductDesignError::InvalidReferenceEvidence(
                "product structural evidence exceeds promotion gate".into(),
            ));
        }
        if gate.require_explicit_modal_residual && self.explicit_modal_relative_residual.is_none() {
            return Err(ProductDesignError::InvalidReferenceEvidence(
                "product structural evidence is missing explicit modal residual".into(),
            ));
        }
        if let Some(residual) = self.explicit_modal_relative_residual
            && (!residual.is_finite()
                || residual < 0.0
                || residual > gate.max_relative_velocity_residual)
        {
            return Err(ProductDesignError::InvalidReferenceEvidence(
                "explicit structural modal residual exceeds promotion gate".into(),
            ));
        }
        Ok(())
    }

    /// Capture reduced structural response together with the residual of the
    /// explicit modal rows solved in the product FEM–BEM matrix.
    pub fn from_monolithic_solution(
        feedback: &StructuralAcousticFeedback,
        solution: &crate::fem_bem_coupling::MonolithicSealedProductSolution,
        pressure_rms_pa: Complex64,
        mesh_convergence_error: f64,
    ) -> Result<Self, ProductDesignError> {
        if solution.structural_generalized_displacement_peak.len() != feedback.modes.len()
            || solution.structural_velocity_peak.len() != feedback.modes.len()
            || !solution.structural_relative_residual.is_finite()
            || solution.structural_relative_residual < 0.0
        {
            return Err(ProductDesignError::InvalidReferenceEvidence(
                "monolithic structural modal evidence has invalid state".into(),
            ));
        }
        if solution
            .structural_generalized_displacement_peak
            .iter()
            .chain(&solution.structural_velocity_peak)
            .any(|value| !value.is_finite())
        {
            return Err(ProductDesignError::InvalidReferenceEvidence(
                "monolithic structural modal state must be finite".into(),
            ));
        }
        let mut evidence = Self::from_feedback(
            feedback,
            solution.frequency_hz,
            pressure_rms_pa,
            1,
            solution.structural_relative_residual,
            mesh_convergence_error,
        )?;
        evidence.explicit_modal_relative_residual = Some(solution.structural_relative_residual);
        evidence.validate_for_point(
            solution.frequency_hz,
            pressure_rms_pa,
            ProductStructuralPromotionGate::default(),
        )?;
        Ok(evidence)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProductReferencePoint {
    pub frequency_hz: f64,
    pub drive_velocity_rms_m_per_s: Complex64,
    /// Per-driver RMS velocities used by the distributed reference solve.
    /// Empty means the legacy scalar field above is the only available
    /// observable.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub driver_velocities_rms_m_per_s: Vec<Complex64>,
    pub aperture_volume_velocity_m3_per_s: Complex64,
    /// Complex mean cavity pressure retained for linear structural loading.
    pub mean_pressure_rms_pa: Complex64,
    pub mean_abs_pressure_pa: f64,
    pub solver_iterations: usize,
    pub relative_residual: f64,
    /// Optional full-product monolithic residual/work/port evidence.  Omitted
    /// by legacy cylindrical reference artifacts; required by promotion gates
    /// that claim a resolved product reference.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub monolithic_evidence: Option<MonolithicProductReferenceEvidence>,
    /// Per-interface conservation evidence retained by resolved product
    /// references. Legacy cylindrical sweeps omit this field and therefore
    /// cannot pass the monolithic promotion gate.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub interface_evidence: Option<Vec<InterfaceReferenceEvidence>>,
    /// Optional per-driver thermal-network evidence for strict product
    /// promotion. Legacy acoustic-only references omit this field.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thermal_evidence: Option<ProductThermalReferenceEvidence>,
    /// Optional structural/acoustic response evidence for strict structural
    /// feedback promotion. Legacy acoustic-only references omit this field.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub structural_evidence: Option<ProductStructuralReferenceEvidence>,
}

/// Residual and conservation evidence for one named FEM/BEM interface.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct InterfaceReferenceEvidence {
    /// Stable interface/patch identifier.
    pub interface_id: String,
    /// Relative block residual for the interface equations.
    pub relative_residual: f64,
    /// Signed power/work residual at the interface.
    pub power_residual_w: f64,
    /// Complex volume flow through the interface, when it is an aperture.
    pub volume_flow_m3_per_s: Complex64,
    /// Optional adjacent-mesh convergence difference.
    pub mesh_convergence_error: Option<f64>,
    /// Optional nonlinear port-flow promotion evidence for this interface.
    ///
    /// This is deliberately attached to the named `port-mouth` interface so
    /// product promotion can retain the complete spatial/temporal evidence
    /// instead of reducing it to one scalar mesh delta.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub port_flow_promotion: Option<PortFlowPromotionEvidence>,
}

impl InterfaceReferenceEvidence {
    fn validate(&self) -> Result<(), ProductDesignError> {
        if self.interface_id.trim().is_empty()
            || !self.relative_residual.is_finite()
            || self.relative_residual < 0.0
            || !self.power_residual_w.is_finite()
            || !self.volume_flow_m3_per_s.re.is_finite()
            || !self.volume_flow_m3_per_s.im.is_finite()
            || self
                .mesh_convergence_error
                .is_some_and(|error| !error.is_finite() || error < 0.0)
        {
            return Err(ProductDesignError::InvalidReferenceEvidence(
                "invalid FEM/BEM interface evidence".into(),
            ));
        }
        if let Some(evidence) = &self.port_flow_promotion {
            if self.interface_id != "port-mouth" {
                return Err(ProductDesignError::InvalidReferenceEvidence(
                    "nonlinear port-flow promotion evidence must belong to port-mouth".into(),
                ));
            }
            validate_port_flow_promotion(evidence, PortFlowPromotionGate::default()).map_err(
                |error| {
                    ProductDesignError::InvalidReferenceEvidence(format!(
                        "invalid nonlinear port-flow promotion evidence: {error}"
                    ))
                },
            )?;
        }
        Ok(())
    }
}

/// Current serialized schema version for independent product validation.
pub const PRODUCT_INDEPENDENT_VALIDATION_SCHEMA_VERSION: u32 = 1;

/// Product observable compared with an independently acquired response.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProductValidationObservable {
    /// Complex mean cavity pressure at the reference point.
    MeanPressureRmsPa,
    /// Complex aggregate drive velocity at the reference point.
    DriveVelocityRmsMPerS,
    /// Complex volume velocity through the resolved aperture.
    ApertureVolumeVelocityM3PerS,
}

/// One independently sourced product-response observation.
///
/// The measured value is intentionally kept separate from the predicted
/// reference point.  Promotion validates that the reported observable and
/// frequency identify the retained product point, and recomputes the error
/// rather than trusting a caller-supplied score alone.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProductIndependentValidationEvidence {
    pub schema_version: u32,
    pub metadata: IndependentMeasurementMetadata,
    pub frequency_hz: f64,
    pub observable: ProductValidationObservable,
    pub measured_value: Complex64,
    pub relative_error: f64,
}

impl ProductIndependentValidationEvidence {
    fn validate_for_point(
        &self,
        point: &ProductReferencePoint,
        gate: &MonolithicEvidenceGate,
    ) -> Result<(), ProductDesignError> {
        if self.schema_version != PRODUCT_INDEPENDENT_VALIDATION_SCHEMA_VERSION {
            return Err(ProductDesignError::InvalidReferenceEvidence(
                "independent product validation schema version is unsupported".into(),
            ));
        }
        self.metadata.validate().map_err(|error| {
            ProductDesignError::InvalidReferenceEvidence(format!(
                "invalid independent product validation metadata: {error}"
            ))
        })?;
        if !self.frequency_hz.is_finite()
            || !self.measured_value.re.is_finite()
            || !self.measured_value.im.is_finite()
            || !self.relative_error.is_finite()
            || self.relative_error < 0.0
        {
            return Err(ProductDesignError::InvalidReferenceEvidence(
                "independent product validation sample must be finite".into(),
            ));
        }
        let frequency_scale = self
            .frequency_hz
            .abs()
            .max(point.frequency_hz.abs())
            .max(1.0);
        if (self.frequency_hz - point.frequency_hz).abs() > 1.0e-9 * frequency_scale {
            return Err(ProductDesignError::InvalidReferenceEvidence(
                "independent product validation frequency differs reference point".into(),
            ));
        }
        let predicted_value = match self.observable {
            ProductValidationObservable::MeanPressureRmsPa => point.mean_pressure_rms_pa,
            ProductValidationObservable::DriveVelocityRmsMPerS => point.drive_velocity_rms_m_per_s,
            ProductValidationObservable::ApertureVolumeVelocityM3PerS => {
                point.aperture_volume_velocity_m3_per_s
            }
        };
        let calculated_error =
            relative_complex_error(self.measured_value, predicted_value, 1.0e-12);
        let error_scale = calculated_error
            .abs()
            .max(self.relative_error.abs())
            .max(1.0e-12);
        if (calculated_error - self.relative_error).abs() > 1.0e-9 * error_scale {
            return Err(ProductDesignError::InvalidReferenceEvidence(
                "independent product validation relative error does not match observable".into(),
            ));
        }
        if self.relative_error > gate.max_independent_validation_error {
            return Err(ProductDesignError::InvalidReferenceEvidence(
                "independent product validation exceeds promotion gate".into(),
            ));
        }
        Ok(())
    }
}

/// Reduced, serializable evidence retained from one monolithic product solve.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MonolithicProductReferenceEvidence {
    pub interior_relative_residual: f64,
    pub exterior_relative_residual: f64,
    pub mechanical_relative_residual: f64,
    pub electrical_relative_residual: f64,
    pub electrical_power_residual_w: f64,
    pub mechanical_power_residual_w: f64,
    pub port_volume_velocity_m3_per_s: Complex64,
    pub maximum_port_pressure_jump_pa: f64,
    pub mesh_node_count: usize,
    pub mesh_tetrahedron_count: usize,
    pub mesh_geometry_fingerprint: String,
    /// Optional externally measured response at this frequency.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub independent_validation: Option<ProductIndependentValidationEvidence>,
}

/// Acceptance limits for an artifact that claims a resolved monolithic
/// product reference rather than a legacy equivalent-cylinder sweep.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct MonolithicEvidenceGate {
    pub max_block_relative_residual: f64,
    pub max_power_residual_w: f64,
    pub max_port_pressure_jump_pa: f64,
    /// Maximum adjacent-mesh observable change accepted for promotion.
    pub max_mesh_convergence_error: f64,
    /// Require complete nonlinear port-flow evidence on every `port-mouth`
    /// interface. Legacy/default gates keep this false for linear fixtures;
    /// strict product promotion should opt in explicitly.
    #[serde(default)]
    pub require_port_flow_promotion: bool,
    /// Require complete per-driver thermal-network evidence for strict
    /// thermoacoustic promotion.
    #[serde(default)]
    pub require_thermal_evidence: bool,
    /// Thermal closure and temperature limits used for product promotion.
    #[serde(default)]
    pub thermal: ProductThermalPromotionGate,
    /// Require retained structural/acoustic response evidence for strict
    /// two-way structural feedback promotion.
    #[serde(default)]
    pub require_structural_evidence: bool,
    /// Structural feedback convergence limits used for product promotion.
    #[serde(default)]
    pub structural: ProductStructuralPromotionGate,
    /// Require a retained three-level product mesh ladder for promotion.
    /// Default/legacy gates keep this false so fixture references remain
    /// readable; product-scale promotion must opt in explicitly.
    #[serde(default)]
    pub require_mesh_convergence_ladder: bool,
    /// Require one independently sourced product response per frequency.
    /// Legacy/default gates keep this false; strict product promotion opts in.
    #[serde(default)]
    pub require_independent_validation: bool,
    /// Maximum independently measured complex-response relative error.
    #[serde(default = "default_max_independent_validation_error")]
    pub max_independent_validation_error: f64,
}

fn default_max_independent_validation_error() -> f64 {
    5.0e-2
}

/// Acceptance limits for promoting a partitioned product result after it has
/// been compared with a resolved monolithic reference on the same frequency
/// grid.  The partitioned path must close its own residual and conservation
/// ledger; agreement with a monolithic observable is not a substitute for
/// those checks.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct PartitionedProductPromotionGate {
    /// Maximum relative residual accepted from the partitioned state or one
    /// of its named interface equations.
    pub max_partitioned_relative_residual: f64,
    /// Maximum signed power/work residual accepted from a partitioned
    /// interface, in watts.
    pub max_partitioned_power_residual_w: f64,
    /// Maximum adjacent-mesh observable difference recorded by a partitioned
    /// interface.
    pub max_partitioned_mesh_convergence_error: f64,
    /// Maximum relative disagreement for retained physical observables.
    pub max_relative_observable_error: f64,
    /// Non-zero scale used when comparing an observable that is physically
    /// zero (for example a sealed-port volume flow).
    pub observable_absolute_floor: f64,
}

impl Default for MonolithicEvidenceGate {
    fn default() -> Self {
        Self {
            max_block_relative_residual: 1.0e-8,
            max_power_residual_w: 1.0e-8,
            max_port_pressure_jump_pa: 1.0e-8,
            max_mesh_convergence_error: 1.0e-2,
            require_port_flow_promotion: false,
            require_thermal_evidence: false,
            thermal: ProductThermalPromotionGate::default(),
            require_structural_evidence: false,
            structural: ProductStructuralPromotionGate::default(),
            require_mesh_convergence_ladder: false,
            require_independent_validation: false,
            max_independent_validation_error: default_max_independent_validation_error(),
        }
    }
}

impl MonolithicEvidenceGate {
    /// Full-product gate for a nonlinear ported reference.
    pub fn strict_nonlinear_product() -> Self {
        Self {
            require_port_flow_promotion: true,
            ..Self::default()
        }
    }

    /// Require an independently sourced product response for every point.
    pub fn strict_independent_product() -> Self {
        Self {
            require_independent_validation: true,
            ..Self::default()
        }
    }

    /// Require both resolved nonlinear port-flow and per-driver thermal
    /// evidence before promoting a product reference.
    pub fn strict_thermoacoustic_product() -> Self {
        Self {
            require_port_flow_promotion: true,
            require_thermal_evidence: true,
            thermal: ProductThermalPromotionGate {
                require_feedback_closure: true,
                ..ProductThermalPromotionGate::default()
            },
            ..Self::default()
        }
    }

    /// Require structural/acoustic feedback evidence in addition to the
    /// nonlinear product contract.
    pub fn strict_structural_product() -> Self {
        Self {
            require_structural_evidence: true,
            structural: ProductStructuralPromotionGate {
                require_explicit_modal_residual: true,
                ..ProductStructuralPromotionGate::default()
            },
            ..Self::default()
        }
    }

    /// Require nonlinear port flow, thermal closure, structural feedback,
    /// and retained product-scale mesh convergence evidence together.
    pub fn strict_full_product_scale() -> Self {
        Self {
            require_port_flow_promotion: true,
            require_thermal_evidence: true,
            require_structural_evidence: true,
            require_mesh_convergence_ladder: true,
            require_independent_validation: true,
            thermal: ProductThermalPromotionGate {
                require_feedback_closure: true,
                ..ProductThermalPromotionGate::default()
            },
            structural: ProductStructuralPromotionGate {
                require_explicit_modal_residual: true,
                ..ProductStructuralPromotionGate::default()
            },
            ..Self::default()
        }
    }

    /// Require a retained, strictly refined product mesh ladder in addition
    /// to the per-point monolithic residual and interface evidence.
    pub fn strict_product_scale() -> Self {
        Self {
            require_mesh_convergence_ladder: true,
            ..Self::default()
        }
    }

    /// Require both product-scale mesh evidence and resolved nonlinear
    /// port-flow evidence for every retained port-mouth reference point.
    ///
    /// The port-flow artifact carries its own spatial and temporal ladders;
    /// coupling it to the product mesh gate prevents a ported product from
    /// being promoted on spatial evidence alone.
    pub fn strict_nonlinear_product_scale() -> Self {
        Self {
            require_port_flow_promotion: true,
            require_mesh_convergence_ladder: true,
            ..Self::default()
        }
    }
}

impl Default for PartitionedProductPromotionGate {
    fn default() -> Self {
        Self {
            max_partitioned_relative_residual: 1.0e-8,
            max_partitioned_power_residual_w: 1.0e-8,
            max_partitioned_mesh_convergence_error: 1.0e-8,
            max_relative_observable_error: 1.0e-4,
            observable_absolute_floor: 1.0e-12,
        }
    }
}

impl MonolithicEvidenceGate {
    fn validate(self) -> Result<(), ProductDesignError> {
        if [
            self.max_block_relative_residual,
            self.max_power_residual_w,
            self.max_port_pressure_jump_pa,
            self.max_mesh_convergence_error,
            self.max_independent_validation_error,
        ]
        .iter()
        .any(|value| !value.is_finite() || *value < 0.0)
        {
            return Err(ProductDesignError::InvalidReferenceEvidence(
                "invalid monolithic evidence gate".into(),
            ));
        }
        self.thermal.validate()?;
        self.structural.validate()?;
        Ok(())
    }
}

impl PartitionedProductPromotionGate {
    fn validate(self) -> Result<(), ProductDesignError> {
        if [
            self.max_partitioned_relative_residual,
            self.max_partitioned_power_residual_w,
            self.max_partitioned_mesh_convergence_error,
            self.max_relative_observable_error,
            self.observable_absolute_floor,
        ]
        .iter()
        .any(|value| !value.is_finite() || *value < 0.0)
        {
            return Err(ProductDesignError::InvalidReferenceEvidence(
                "invalid partitioned product promotion gate".into(),
            ));
        }
        if self.observable_absolute_floor == 0.0 {
            return Err(ProductDesignError::InvalidReferenceEvidence(
                "partitioned observable comparison floor must be positive".into(),
            ));
        }
        Ok(())
    }
}

fn relative_scalar_error(actual: f64, reference: f64, floor: f64) -> f64 {
    (actual - reference).abs() / actual.abs().max(reference.abs()).max(floor)
}

fn relative_complex_error(actual: Complex64, reference: Complex64, floor: f64) -> f64 {
    (actual - reference).norm() / actual.norm().max(reference.norm()).max(floor)
}

impl MonolithicProductReferenceEvidence {
    /// Capture the validated reduced evidence without serializing full field
    /// vectors.  Mesh identity is retained so convergence gates can compare
    /// like-for-like product topologies.
    pub fn from_monolithic_solution(
        solution: &crate::fem_bem_coupling::MonolithicSealedProductSolution,
        mesh: &crate::product_acoustic_mesh::ProductInteriorMesh,
        mesh_convergence_error: f64,
    ) -> Result<Self, ProductDesignError> {
        if !mesh_convergence_error.is_finite() || mesh_convergence_error < 0.0 {
            return Err(ProductDesignError::InvalidReferenceEvidence(
                "monolithic mesh-convergence evidence must be finite and non-negative".into(),
            ));
        }
        Ok(Self {
            interior_relative_residual: solution.interior_relative_residual,
            exterior_relative_residual: solution.exterior_relative_residual,
            mechanical_relative_residual: solution.mechanical_relative_residual,
            electrical_relative_residual: solution.electrical_relative_residual,
            electrical_power_residual_w: solution.electrical_power_residual_w,
            mechanical_power_residual_w: solution.mechanical_power_residual_w,
            port_volume_velocity_m3_per_s: solution.port_volume_velocity_m3_per_s,
            maximum_port_pressure_jump_pa: solution
                .port_pressure_jump_pa
                .iter()
                .map(|jump| jump.norm())
                .fold(0.0, f64::max),
            mesh_node_count: mesh.diagnostics.node_count,
            mesh_tetrahedron_count: mesh.diagnostics.tetrahedron_count,
            mesh_geometry_fingerprint: mesh.geometry_fingerprint.clone(),
            independent_validation: None,
        })
    }

    fn validate(&self) -> Result<(), ProductDesignError> {
        let finite = [
            self.interior_relative_residual,
            self.exterior_relative_residual,
            self.mechanical_relative_residual,
            self.electrical_relative_residual,
            self.electrical_power_residual_w,
            self.mechanical_power_residual_w,
            self.port_volume_velocity_m3_per_s.re,
            self.port_volume_velocity_m3_per_s.im,
            self.maximum_port_pressure_jump_pa,
        ];
        if finite.iter().any(|value| !value.is_finite())
            || self.interior_relative_residual < 0.0
            || self.exterior_relative_residual < 0.0
            || self.mechanical_relative_residual < 0.0
            || self.electrical_relative_residual < 0.0
            || self.maximum_port_pressure_jump_pa < 0.0
            || self.mesh_node_count == 0
            || self.mesh_tetrahedron_count == 0
            || self.mesh_geometry_fingerprint.trim().is_empty()
        {
            return Err(ProductDesignError::InvalidReferenceEvidence(
                "invalid monolithic product reference evidence".into(),
            ));
        }
        Ok(())
    }
}

impl ProductReferencePoint {
    /// Convert a monolithic solve without optional explicit structural
    /// modal evidence.
    pub fn from_monolithic_solution(
        solution: &crate::fem_bem_coupling::MonolithicSealedProductSolution,
        mesh: &crate::product_acoustic_mesh::ProductInteriorMesh,
        mesh_convergence_error: f64,
    ) -> Result<Self, ProductDesignError> {
        Self::from_monolithic_solution_with_structural_feedback(
            solution,
            mesh,
            mesh_convergence_error,
            None,
        )
    }
    /// Convert one full-product monolithic solve into the portable reference
    /// artifact consumed by promotion gates.  The legacy scalar drive field is
    /// the first driver for compatibility; the per-driver vector remains the
    /// authoritative product result.
    pub fn from_monolithic_solution_with_structural_feedback(
        solution: &crate::fem_bem_coupling::MonolithicSealedProductSolution,
        mesh: &crate::product_acoustic_mesh::ProductInteriorMesh,
        mesh_convergence_error: f64,
        structural_feedback: Option<&StructuralAcousticFeedback>,
    ) -> Result<Self, ProductDesignError> {
        if !mesh_convergence_error.is_finite() || mesh_convergence_error < 0.0 {
            return Err(ProductDesignError::InvalidReferenceEvidence(
                "monolithic mesh-convergence evidence must be finite and non-negative".into(),
            ));
        }
        let mean_pressure_rms_pa = if solution.interior_pressure.is_empty() {
            Complex64::default()
        } else {
            solution
                .interior_pressure
                .iter()
                .copied()
                .sum::<Complex64>()
                / solution.interior_pressure.len() as f64
        };
        let driver_count = solution.driver_velocity_m_per_s.len();
        if solution.driver_mechanical_relative_residual.len() != driver_count
            || solution.driver_electrical_relative_residual.len() != driver_count
            || solution.driver_volume_velocity_m3_per_s.len() != driver_count
            || solution.driver_electrical_power_residual_w.len() != driver_count
            || solution.driver_mechanical_power_residual_w.len() != driver_count
        {
            return Err(ProductDesignError::InvalidReferenceEvidence(
                "monolithic driver interface evidence length mismatch".into(),
            ));
        }
        let mut interface_evidence = Vec::with_capacity(
            driver_count + usize::from(!solution.port_velocity_m_per_s.is_empty()),
        );
        if !solution.port_velocity_m_per_s.is_empty() {
            interface_evidence.push(InterfaceReferenceEvidence {
                interface_id: "port-mouth".into(),
                relative_residual: solution.relative_residual,
                power_residual_w: solution.mechanical_power_residual_w,
                volume_flow_m3_per_s: solution.port_volume_velocity_m3_per_s,
                mesh_convergence_error: Some(mesh_convergence_error),
                port_flow_promotion: None,
            });
        }
        interface_evidence.extend((0..driver_count).map(|driver_index| {
            InterfaceReferenceEvidence {
                interface_id: format!("driver-aperture-{driver_index}"),
                relative_residual: solution.driver_mechanical_relative_residual[driver_index]
                    .max(solution.driver_electrical_relative_residual[driver_index]),
                power_residual_w: solution.driver_electrical_power_residual_w[driver_index]
                    + solution.driver_mechanical_power_residual_w[driver_index],
                volume_flow_m3_per_s: solution.driver_volume_velocity_m3_per_s[driver_index],
                mesh_convergence_error: Some(mesh_convergence_error),
                port_flow_promotion: None,
            }
        }));
        let mut point = Self {
            frequency_hz: solution.frequency_hz,
            drive_velocity_rms_m_per_s: solution
                .driver_velocity_m_per_s
                .first()
                .copied()
                .unwrap_or_default(),
            driver_velocities_rms_m_per_s: solution.driver_velocity_m_per_s.clone(),
            aperture_volume_velocity_m3_per_s: solution.port_volume_velocity_m3_per_s,
            mean_pressure_rms_pa,
            mean_abs_pressure_pa: solution
                .interior_pressure
                .iter()
                .map(|pressure| pressure.norm())
                .sum::<f64>()
                / solution.interior_pressure.len().max(1) as f64,
            solver_iterations: 1,
            relative_residual: solution.relative_residual,
            monolithic_evidence: Some(
                MonolithicProductReferenceEvidence::from_monolithic_solution(
                    solution,
                    mesh,
                    mesh_convergence_error,
                )?,
            ),
            interface_evidence: Some(interface_evidence),
            thermal_evidence: None,
            structural_evidence: None,
        };
        if let Some(feedback) = structural_feedback {
            point.structural_evidence = Some(
                ProductStructuralReferenceEvidence::from_monolithic_solution(
                    feedback,
                    solution,
                    mean_pressure_rms_pa,
                    mesh_convergence_error,
                )?,
            );
        }
        point.validate()?;
        Ok(point)
    }

    fn validate(&self) -> Result<(), ProductDesignError> {
        if !self.frequency_hz.is_finite() || self.frequency_hz <= 0.0 {
            return Err(ProductDesignError::InvalidFrequency);
        }
        for (field, value) in [
            ("drive velocity real", self.drive_velocity_rms_m_per_s.re),
            (
                "drive velocity imaginary",
                self.drive_velocity_rms_m_per_s.im,
            ),
            (
                "aperture volume velocity real",
                self.aperture_volume_velocity_m3_per_s.re,
            ),
            (
                "aperture volume velocity imaginary",
                self.aperture_volume_velocity_m3_per_s.im,
            ),
            ("mean pressure real", self.mean_pressure_rms_pa.re),
            ("mean pressure imaginary", self.mean_pressure_rms_pa.im),
            ("mean absolute pressure", self.mean_abs_pressure_pa),
            ("relative residual", self.relative_residual),
        ] {
            if !value.is_finite() {
                return Err(ProductDesignError::InvalidReferenceEvidence(
                    field.to_owned(),
                ));
            }
        }
        if self
            .driver_velocities_rms_m_per_s
            .iter()
            .any(|value| !value.re.is_finite() || !value.im.is_finite())
        {
            return Err(ProductDesignError::InvalidReferenceEvidence(
                "distributed driver velocity must be finite".into(),
            ));
        }
        if let Some(&first_driver_velocity) = self.driver_velocities_rms_m_per_s.first() {
            let mismatch = (first_driver_velocity - self.drive_velocity_rms_m_per_s).norm();
            let scale = first_driver_velocity
                .norm()
                .max(self.drive_velocity_rms_m_per_s.norm())
                .max(1.0e-30);
            if mismatch > 1.0e-12 * scale {
                return Err(ProductDesignError::InvalidReferenceEvidence(
                    "legacy scalar drive velocity differs from first distributed driver".into(),
                ));
            }
        }
        if self.mean_abs_pressure_pa < 0.0 || self.relative_residual < 0.0 {
            return Err(ProductDesignError::InvalidReferenceEvidence(
                "reference observables must be non-negative".into(),
            ));
        }
        if let Some(evidence) = &self.monolithic_evidence {
            evidence.validate()?;
        }
        if let Some(evidence) = &self.interface_evidence {
            if evidence.is_empty() {
                return Err(ProductDesignError::InvalidReferenceEvidence(
                    "interface evidence must not be empty".into(),
                ));
            }
            let mut ids = BTreeSet::new();
            for interface in evidence {
                interface.validate()?;
                if !ids.insert(interface.interface_id.as_str()) {
                    return Err(ProductDesignError::InvalidReferenceEvidence(
                        "interface evidence contains duplicate interface ids".into(),
                    ));
                }
            }
        }
        if let Some(evidence) = &self.thermal_evidence {
            evidence.validate_for_driver_count(
                self.driver_velocities_rms_m_per_s.len(),
                ProductThermalPromotionGate::default(),
            )?;
        }
        if let Some(evidence) = &self.structural_evidence {
            evidence.validate_for_point(
                self.frequency_hz,
                self.mean_pressure_rms_pa,
                ProductStructuralPromotionGate::default(),
            )?;
        }
        Ok(())
    }
}

fn retained_product_observable_error(
    coarser: &ProductReferencePoint,
    finer: &ProductReferencePoint,
) -> Result<f64, ProductDesignError> {
    coarser.validate()?;
    finer.validate()?;
    let frequency_scale = coarser.frequency_hz.max(finer.frequency_hz).max(1.0);
    if (coarser.frequency_hz - finer.frequency_hz).abs() > 1.0e-9 * frequency_scale {
        return Err(ProductDesignError::InvalidReferenceEvidence(
            "adjacent mesh points must use the same finite frequency".into(),
        ));
    }
    if coarser.driver_velocities_rms_m_per_s.len() != finer.driver_velocities_rms_m_per_s.len() {
        return Err(ProductDesignError::InvalidReferenceEvidence(
            "adjacent mesh points must retain the same driver count".into(),
        ));
    }

    let mut maximum = relative_complex_error(
        finer.drive_velocity_rms_m_per_s,
        coarser.drive_velocity_rms_m_per_s,
        1.0e-30,
    )
    .max(relative_complex_error(
        finer.aperture_volume_velocity_m3_per_s,
        coarser.aperture_volume_velocity_m3_per_s,
        1.0e-30,
    ))
    .max(relative_complex_error(
        finer.mean_pressure_rms_pa,
        coarser.mean_pressure_rms_pa,
        1.0e-30,
    ))
    .max(relative_scalar_error(
        finer.mean_abs_pressure_pa,
        coarser.mean_abs_pressure_pa,
        1.0e-30,
    ));
    for (coarser_velocity, finer_velocity) in coarser
        .driver_velocities_rms_m_per_s
        .iter()
        .zip(&finer.driver_velocities_rms_m_per_s)
    {
        maximum = maximum.max(relative_complex_error(
            *finer_velocity,
            *coarser_velocity,
            1.0e-30,
        ));
    }

    match (&coarser.interface_evidence, &finer.interface_evidence) {
        (Some(coarser_interfaces), Some(finer_interfaces)) => {
            if coarser_interfaces.len() != finer_interfaces.len()
                || coarser_interfaces
                    .iter()
                    .zip(finer_interfaces)
                    .any(|(coarser, finer)| coarser.interface_id != finer.interface_id)
            {
                return Err(ProductDesignError::InvalidReferenceEvidence(
                    "adjacent mesh points must retain identical interface identities".into(),
                ));
            }
            for (coarser, finer) in coarser_interfaces.iter().zip(finer_interfaces) {
                maximum = maximum.max(relative_complex_error(
                    finer.volume_flow_m3_per_s,
                    coarser.volume_flow_m3_per_s,
                    1.0e-30,
                ));
            }
        }
        (None, None) => {}
        _ => {
            return Err(ProductDesignError::InvalidReferenceEvidence(
                "adjacent mesh points must both retain interface evidence".into(),
            ));
        }
    }

    match (&coarser.thermal_evidence, &finer.thermal_evidence) {
        (Some(coarser_thermal), Some(finer_thermal)) => {
            if coarser_thermal.driver_ids != finer_thermal.driver_ids
                || coarser_thermal.source_power_w.len() != finer_thermal.source_power_w.len()
                || coarser_thermal.states.len() != finer_thermal.states.len()
            {
                return Err(ProductDesignError::InvalidReferenceEvidence(
                    "adjacent mesh points must retain identical thermal identities".into(),
                ));
            }
            for (coarser_power, finer_power) in coarser_thermal
                .source_power_w
                .iter()
                .zip(&finer_thermal.source_power_w)
            {
                for (coarser_power, finer_power) in [
                    (coarser_power.voice_coil_w, finer_power.voice_coil_w),
                    (coarser_power.magnet_w, finer_power.magnet_w),
                    (coarser_power.gap_w, finer_power.gap_w),
                    (coarser_power.cabinet_w, finer_power.cabinet_w),
                ] {
                    maximum =
                        maximum.max(relative_scalar_error(finer_power, coarser_power, 1.0e-30));
                }
            }
            for (coarser_state, finer_state) in
                coarser_thermal.states.iter().zip(&finer_thermal.states)
            {
                for (coarser_temperature, finer_temperature) in coarser_state
                    .temperatures_k
                    .iter()
                    .zip(&finer_state.temperatures_k)
                {
                    maximum = maximum.max(relative_scalar_error(
                        *finer_temperature,
                        *coarser_temperature,
                        1.0e-30,
                    ));
                }
            }
        }
        (None, None) => {}
        _ => {
            return Err(ProductDesignError::InvalidReferenceEvidence(
                "adjacent mesh points must both retain thermal evidence".into(),
            ));
        }
    }

    if !maximum.is_finite() {
        return Err(ProductDesignError::InvalidReferenceEvidence(
            "adjacent mesh observable change is non-finite".into(),
        ));
    }
    Ok(maximum)
}

/// Compute the maximum normalized change across all retained product
/// observables between adjacent meshes. This is independent of algebraic
/// residual budgets and is suitable for interface evidence.
pub fn adjacent_mesh_convergence_error(
    coarser: &ProductReferencePoint,
    finer: &ProductReferencePoint,
) -> Result<f64, ProductDesignError> {
    match (&coarser.monolithic_evidence, &finer.monolithic_evidence) {
        (Some(coarser_mesh), Some(finer_mesh)) => {
            if coarser_mesh.mesh_geometry_fingerprint == finer_mesh.mesh_geometry_fingerprint
                || finer_mesh.mesh_node_count <= coarser_mesh.mesh_node_count
                || finer_mesh.mesh_tetrahedron_count <= coarser_mesh.mesh_tetrahedron_count
            {
                return Err(ProductDesignError::InvalidReferenceEvidence(
                    "adjacent mesh points must represent a strictly refined mesh".into(),
                ));
            }
        }
        (None, None) => {}
        _ => {
            return Err(ProductDesignError::InvalidReferenceEvidence(
                "adjacent mesh points must both retain monolithic mesh identity".into(),
            ));
        }
    }
    retained_product_observable_error(coarser, finer)
}

/// Current serialized schema version for the product mesh-convergence
/// ladder. A ladder is deliberately separate from a single point's optional
/// adjacent-mesh scalar so promotion can require three retained levels.
/// Input specification for one independently solved body-fitted product
/// mesh in a convergence ladder.
///
/// The characteristic size is taken from `mesh_config.max_element_size_m`
/// so retained evidence cannot drift from the requested meshing policy.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProductBodyFittedMeshLevel {
    /// Stable human-readable identity retained in each ladder level.
    pub label: String,
    /// Independent body-fitted mesh policy used for this level.
    pub mesh_config: ProductInteriorMeshConfig,
}

pub const PRODUCT_MESH_CONVERGENCE_SCHEMA_VERSION: u32 = 1;

/// One physical product reference point at a declared mesh level.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProductMeshConvergenceLevel {
    pub label: String,
    /// Declared characteristic target size; strictly decreases toward fine.
    pub characteristic_size_m: f64,
    pub point: ProductReferencePoint,
}

/// Versioned, serializable product-scale mesh-convergence evidence.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ProductMeshConvergenceEvidence {
    pub schema_version: u32,
    pub frequency_hz: f64,
    pub levels: Vec<ProductMeshConvergenceLevel>,
    pub adjacent_observable_errors: Vec<f64>,
    pub maximum_adjacent_observable_error: f64,
}

#[derive(Debug, Deserialize)]
struct ProductMeshConvergenceEvidenceWire {
    schema_version: u32,
    frequency_hz: f64,
    levels: Vec<ProductMeshConvergenceLevel>,
    adjacent_observable_errors: Vec<f64>,
    maximum_adjacent_observable_error: f64,
}

impl<'de> Deserialize<'de> for ProductMeshConvergenceEvidence {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let wire = ProductMeshConvergenceEvidenceWire::deserialize(deserializer)?;
        let evidence = Self {
            schema_version: wire.schema_version,
            frequency_hz: wire.frequency_hz,
            levels: wire.levels,
            adjacent_observable_errors: wire.adjacent_observable_errors,
            maximum_adjacent_observable_error: wire.maximum_adjacent_observable_error,
        };
        evidence
            .validate()
            .map_err(|error| <D::Error as serde::de::Error>::custom(error.to_string()))?;
        Ok(evidence)
    }
}

impl ProductMeshConvergenceEvidence {
    /// Build and derive the retained adjacent-observable evidence from at
    /// least three independently solved, strictly refined levels.
    pub fn from_levels(
        levels: Vec<ProductMeshConvergenceLevel>,
    ) -> Result<Self, ProductDesignError> {
        let frequency_hz = levels
            .first()
            .map(|level| level.point.frequency_hz)
            .unwrap_or(f64::NAN);
        let evidence = Self {
            schema_version: PRODUCT_MESH_CONVERGENCE_SCHEMA_VERSION,
            frequency_hz,
            levels,
            adjacent_observable_errors: Vec::new(),
            maximum_adjacent_observable_error: f64::NAN,
        };
        evidence.recompute()
    }

    fn recompute(mut self) -> Result<Self, ProductDesignError> {
        self.validate_shape()?;
        self.adjacent_observable_errors = self
            .levels
            .windows(2)
            .map(|window| adjacent_mesh_convergence_error(&window[0].point, &window[1].point))
            .collect::<Result<Vec<_>, _>>()?;
        self.maximum_adjacent_observable_error = self
            .adjacent_observable_errors
            .iter()
            .copied()
            .fold(0.0, f64::max);
        Ok(self)
    }

    fn validate_shape(&self) -> Result<(), ProductDesignError> {
        if self.schema_version != PRODUCT_MESH_CONVERGENCE_SCHEMA_VERSION
            || !self.frequency_hz.is_finite()
            || self.frequency_hz <= 0.0
            || self.levels.len() < 3
        {
            return Err(ProductDesignError::InvalidReferenceEvidence(
                "product mesh convergence requires schema v1, a positive frequency, and at least three levels".into(),
            ));
        }
        let mut labels = BTreeSet::new();
        for (index, level) in self.levels.iter().enumerate() {
            if level.label.trim().is_empty()
                || !labels.insert(level.label.as_str())
                || !level.characteristic_size_m.is_finite()
                || level.characteristic_size_m <= 0.0
                || level.point.frequency_hz != self.frequency_hz
            {
                return Err(ProductDesignError::InvalidReferenceEvidence(
                    "invalid product mesh convergence level metadata".into(),
                ));
            }
            if index > 0
                && level.characteristic_size_m >= self.levels[index - 1].characteristic_size_m
            {
                return Err(ProductDesignError::InvalidReferenceEvidence(
                    "product mesh convergence characteristic sizes must strictly refine".into(),
                ));
            }
            level.point.validate()?;
        }
        Ok(())
    }

    pub fn validate(&self) -> Result<(), ProductDesignError> {
        self.validate_shape()?;
        if self.adjacent_observable_errors.len() != self.levels.len() - 1
            || self
                .adjacent_observable_errors
                .iter()
                .any(|error| !error.is_finite() || *error < 0.0)
            || !self.maximum_adjacent_observable_error.is_finite()
            || self.maximum_adjacent_observable_error < 0.0
        {
            return Err(ProductDesignError::InvalidReferenceEvidence(
                "invalid product mesh convergence observable evidence".into(),
            ));
        }
        let recomputed = self
            .levels
            .windows(2)
            .map(|window| adjacent_mesh_convergence_error(&window[0].point, &window[1].point))
            .collect::<Result<Vec<_>, _>>()?;
        if recomputed
            .iter()
            .zip(&self.adjacent_observable_errors)
            .any(|(expected, actual)| {
                (expected - actual).abs() > 1.0e-12 * expected.abs().max(actual.abs()).max(1.0)
            })
            || (recomputed.iter().copied().fold(0.0, f64::max)
                - self.maximum_adjacent_observable_error)
                .abs()
                > 1.0e-12 * self.maximum_adjacent_observable_error.max(1.0)
        {
            return Err(ProductDesignError::InvalidReferenceEvidence(
                "product mesh convergence observable evidence does not match retained levels"
                    .into(),
            ));
        }
        Ok(())
    }

    pub fn finest_point(&self) -> Option<&ProductReferencePoint> {
        self.levels.last().map(|level| &level.point)
    }
}

/// Product-facing evidence from the distributed FEM–BEM reference sweep.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ProductReferenceSweep {
    pub speed_of_sound_m_per_s: f64,
    pub density_kg_per_m3: f64,
    pub interior_dofs: usize,
    pub aperture_elements: usize,
    pub points: Vec<ProductReferencePoint>,
    /// Stable product driver count used by the distributed reference points.
    /// A zero value is accepted only for legacy serialized artifacts and is
    /// interpreted as one driver.
    #[serde(default)]
    pub driver_count: usize,
    /// Portable artifact provenance for the retained full-order reference.
    pub provenance: ArtifactProvenance,
    /// Optional residual-gated reduced-order evidence for the same coupling
    /// fixture and frequency grid.  The physical reference observables above
    /// remain full-order; this summary records whether a ROM point was
    /// accepted or promoted before product consumers use the sweep.
    pub rom_promotion: Option<ProductRomPromotionEvidence>,
    /// Optional per-frequency product-scale mesh ladders. Each retained
    /// ladder must contain at least three strictly refined levels.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub mesh_convergence: Vec<ProductMeshConvergenceEvidence>,
}

/// Wire representation for retained reference evidence crossing a serde
/// boundary. Reduced observables are only meaningful when the full sweep
/// contract holds, so deserialization must not leave that validation optional.
#[derive(Debug, Deserialize)]
struct ProductReferenceSweepWire {
    speed_of_sound_m_per_s: f64,
    density_kg_per_m3: f64,
    interior_dofs: usize,
    aperture_elements: usize,
    points: Vec<ProductReferencePoint>,
    #[serde(default)]
    driver_count: usize,
    provenance: ArtifactProvenance,
    rom_promotion: Option<ProductRomPromotionEvidence>,
    #[serde(default)]
    mesh_convergence: Vec<ProductMeshConvergenceEvidence>,
}

impl<'de> Deserialize<'de> for ProductReferenceSweep {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let wire = ProductReferenceSweepWire::deserialize(deserializer)?;
        let sweep = Self {
            speed_of_sound_m_per_s: wire.speed_of_sound_m_per_s,
            density_kg_per_m3: wire.density_kg_per_m3,
            interior_dofs: wire.interior_dofs,
            aperture_elements: wire.aperture_elements,
            points: wire.points,
            driver_count: wire.driver_count,
            provenance: wire.provenance,
            rom_promotion: wire.rom_promotion,
            mesh_convergence: wire.mesh_convergence,
        };
        sweep
            .validate()
            .map_err(|error| <D::Error as serde::de::Error>::custom(error.to_string()))?;
        Ok(sweep)
    }
}

/// Product-facing summary of an adaptive FEM–BEM ROM sweep.
///
/// The full [`RomPromotionReport`] intentionally stays inside the modeling
/// coupling module.  Product artifacts only need the typed gate outcomes and
/// counts, which keeps the simulator/modeling boundary stable and avoids
/// exposing reduced basis internals to the speaker application.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProductRomPromotionEvidence {
    pub point_count: usize,
    pub promoted_count: usize,
    pub maximum_checked_reduced_residual: Option<f64>,
    pub minimum_checked_rank_agreement: Option<f64>,
    pub maximum_checked_gradient_relative_error: Option<f64>,
    pub maximum_checked_constraint_violation: Option<f64>,
}

impl ProductRomPromotionEvidence {
    fn from_report(report: &RomPromotionReport) -> Self {
        Self {
            point_count: report.points.len(),
            promoted_count: report.promoted_count,
            maximum_checked_reduced_residual: report.maximum_checked_reduced_residual,
            minimum_checked_rank_agreement: report.minimum_checked_rank_agreement,
            maximum_checked_gradient_relative_error: report.maximum_checked_gradient_relative_error,
            maximum_checked_constraint_violation: report.maximum_checked_constraint_violation,
        }
    }

    fn validate(&self) -> Result<(), ProductDesignError> {
        if self.point_count == 0 || self.promoted_count > self.point_count {
            return Err(ProductDesignError::InvalidReferenceEvidence(
                "ROM promotion counts are inconsistent".into(),
            ));
        }
        if let Some(value) = self.maximum_checked_reduced_residual
            && (!value.is_finite() || value < 0.0)
        {
            return Err(ProductDesignError::InvalidReferenceEvidence(
                "ROM reduced residual must be finite and non-negative".into(),
            ));
        }
        if let Some(value) = self.minimum_checked_rank_agreement
            && (!value.is_finite() || !(0.0..=1.0).contains(&value))
        {
            return Err(ProductDesignError::InvalidReferenceEvidence(
                "ROM rank agreement must be finite and in [0, 1]".into(),
            ));
        }
        for value in [
            self.maximum_checked_gradient_relative_error,
            self.maximum_checked_constraint_violation,
        ]
        .into_iter()
        .flatten()
        {
            if !value.is_finite() || value < 0.0 {
                return Err(ProductDesignError::InvalidReferenceEvidence(
                    "ROM diagnostics must be finite and non-negative".into(),
                ));
            }
        }
        Ok(())
    }
}

impl ProductReferenceSweep {
    pub fn validate(&self) -> Result<(), ProductDesignError> {
        if !self.speed_of_sound_m_per_s.is_finite() || self.speed_of_sound_m_per_s <= 0.0 {
            return Err(ProductDesignError::InvalidReferenceEvidence(
                "speed of sound must be finite and positive".into(),
            ));
        }
        if !self.density_kg_per_m3.is_finite() || self.density_kg_per_m3 <= 0.0 {
            return Err(ProductDesignError::InvalidReferenceEvidence(
                "density must be finite and positive".into(),
            ));
        }
        if self.interior_dofs == 0 || self.aperture_elements == 0 || self.points.is_empty() {
            return Err(ProductDesignError::InvalidReferenceEvidence(
                "reference sweep must contain a non-empty distributed model".into(),
            ));
        }
        let driver_count = self.driver_count.max(1);
        let has_distributed_driver_evidence = self
            .points
            .iter()
            .any(|point| !point.driver_velocities_rms_m_per_s.is_empty());
        for point in &self.points {
            if has_distributed_driver_evidence && point.driver_velocities_rms_m_per_s.is_empty() {
                return Err(ProductDesignError::InvalidReferenceEvidence(
                    "distributed reference driver evidence must be present at every frequency"
                        .into(),
                ));
            }
            if !point.driver_velocities_rms_m_per_s.is_empty()
                && point.driver_velocities_rms_m_per_s.len() != driver_count
            {
                return Err(ProductDesignError::InvalidReferenceEvidence(
                    "distributed reference driver count differs from sweep metadata".into(),
                ));
            }
        }
        if self
            .points
            .windows(2)
            .any(|window| window[1].frequency_hz <= window[0].frequency_hz)
        {
            return Err(ProductDesignError::InvalidReferenceEvidence(
                "reference sweep frequencies must be strictly increasing".into(),
            ));
        }
        for point in &self.points {
            point.validate()?;
        }
        if let Some(rom) = &self.rom_promotion {
            rom.validate()?;
            if rom.point_count != self.points.len() {
                return Err(ProductDesignError::InvalidReferenceEvidence(
                    "ROM promotion and reference sweep grids differ".into(),
                ));
            }
        }
        let mut mesh_frequencies = BTreeSet::new();
        for ladder in &self.mesh_convergence {
            ladder.validate()?;
            if !mesh_frequencies.insert(ladder.frequency_hz.to_bits()) {
                return Err(ProductDesignError::InvalidReferenceEvidence(
                    "reference sweep contains duplicate mesh-convergence frequencies".into(),
                ));
            }
            if !self
                .points
                .iter()
                .any(|point| point.frequency_hz == ladder.frequency_hz)
            {
                return Err(ProductDesignError::InvalidReferenceEvidence(
                    "mesh-convergence ladder frequency is absent from reference sweep".into(),
                ));
            }
        }
        Ok(())
    }

    /// Validate that this retained reference sweep belongs to the canonical
    /// product coupling mesh supplied by the caller.
    ///
    /// The monolithic point evidence stores the interior mesh fingerprint,
    /// while [`ProductCouplingMesh`] additionally owns the exterior surface
    /// and typed FEM/BEM traces. Binding both here prevents a valid-looking
    /// point from being promoted after its geometry, driver count, interface
    /// set, or FEM topology has changed. This is an artifact/promotion gate;
    /// it does not execute a solve or establish physical convergence.
    pub fn validate_against_product_coupling_mesh(
        &self,
        coupling_mesh: &ProductCouplingMesh,
    ) -> Result<(), ProductDesignError> {
        self.validate()?;
        coupling_mesh
            .validate_evidence(&coupling_mesh.evidence())
            .map_err(|error| {
                ProductDesignError::InvalidReferenceEvidence(format!(
                    "canonical product coupling mesh evidence is invalid: {error}"
                ))
            })?;

        let mesh_driver_count = coupling_mesh
            .traces
            .keys()
            .filter(|patch| matches!(patch, ProductInteriorPatchKind::DriverRear { .. }))
            .count();
        if self.driver_count.max(1) != mesh_driver_count {
            return Err(ProductDesignError::InvalidReferenceEvidence(
                "reference sweep driver count differs from canonical product coupling mesh".into(),
            ));
        }
        if self.interior_dofs != coupling_mesh.interior.diagnostics.node_count {
            return Err(ProductDesignError::InvalidReferenceEvidence(
                "reference sweep interior DOF count differs from canonical product coupling mesh"
                    .into(),
            ));
        }

        let mut required_interface_ids = BTreeSet::new();
        for patch in coupling_mesh.traces.keys() {
            match patch {
                ProductInteriorPatchKind::DriverRear { driver_index } => {
                    required_interface_ids.insert(format!("driver-aperture-{driver_index}"));
                }
                ProductInteriorPatchKind::PortMouth => {
                    required_interface_ids.insert("port-mouth".to_owned());
                }
                ProductInteriorPatchKind::RigidWall(_)
                | ProductInteriorPatchKind::WaveguideWall { .. }
                | ProductInteriorPatchKind::PortWall
                | ProductInteriorPatchKind::BraceWall { .. } => {}
            }
        }

        for (point_index, point) in self.points.iter().enumerate() {
            let monolithic = point.monolithic_evidence.as_ref().ok_or_else(|| {
                ProductDesignError::InvalidReferenceEvidence(format!(
                    "reference point {point_index} is missing monolithic evidence for canonical mesh binding"
                ))
            })?;
            if monolithic.mesh_node_count != coupling_mesh.interior.diagnostics.node_count
                || monolithic.mesh_tetrahedron_count
                    != coupling_mesh.interior.diagnostics.tetrahedron_count
            {
                return Err(ProductDesignError::InvalidReferenceEvidence(format!(
                    "reference point {point_index} FEM topology differs from canonical product coupling mesh"
                )));
            }
            if monolithic.mesh_geometry_fingerprint != coupling_mesh.interior.geometry_fingerprint {
                return Err(ProductDesignError::InvalidReferenceEvidence(format!(
                    "reference point {point_index} mesh fingerprint differs from canonical product coupling mesh"
                )));
            }

            let interfaces = point.interface_evidence.as_ref().ok_or_else(|| {
                ProductDesignError::InvalidReferenceEvidence(format!(
                    "reference point {point_index} is missing interface evidence for canonical mesh binding"
                ))
        })?;
            for required_id in &required_interface_ids {
                if !interfaces
                    .iter()
                    .any(|interface| interface.interface_id == *required_id)
                {
                    return Err(ProductDesignError::InvalidReferenceEvidence(format!(
                        "reference point {point_index} is missing canonical interface {required_id}"
                    )));
                }
            }

            let observed_interface_ids = interfaces
                .iter()
                .map(|interface| interface.interface_id.clone())
                .collect::<BTreeSet<_>>();
            if observed_interface_ids != required_interface_ids {
                return Err(ProductDesignError::InvalidReferenceEvidence(format!(
                    "reference point {point_index} interface identities differ from canonical product coupling mesh"
                )));
            }
        }
        Ok(())
    }

    /// Attach solver-free replay evidence from the executable port-flow
    /// promotion runner to the matching product reference points.
    ///
    /// All artifact, frequency, interface, and existing-evidence checks are
    /// completed before any point is changed. The runtime artifact is only
    /// replayed here; this adapter does not run a FEM solve or establish
    /// physical convergence beyond the evidence gate supplied to replay.
    pub fn attach_port_flow_runtime_promotions(
        &mut self,
        artifacts: &[PortFlowRuntimePromotionArtifact],
    ) -> Result<(), ProductDesignError> {
        self.validate()?;

        if artifacts.len() != self.points.len() {
            return Err(ProductDesignError::InvalidReferenceEvidence(format!(
                "port-flow runtime artifact count {} differs product reference point count {}",
                artifacts.len(),
                self.points.len(),
            )));
        }

        let mut replayed_evidence = Vec::with_capacity(artifacts.len());
        for (point_index, (point, artifact)) in self.points.iter().zip(artifacts).enumerate() {
            let interfaces = point.interface_evidence.as_ref().ok_or_else(|| {
                ProductDesignError::InvalidReferenceEvidence(format!(
                    "product reference point {point_index} missing interface evidence for port-flow runtime promotion"
                ))
            })?;
            let port_mouth_indices = interfaces
                .iter()
                .enumerate()
                .filter_map(|(interface_index, interface)| {
                    (interface.interface_id == "port-mouth").then_some(interface_index)
                })
                .collect::<Vec<_>>();
            if port_mouth_indices.len() != 1 {
                return Err(ProductDesignError::InvalidReferenceEvidence(format!(
                    "product reference point {point_index} must contain exactly one port-mouth interface, found {}",
                    port_mouth_indices.len(),
                )));
            }
            let port_mouth = &interfaces[port_mouth_indices[0]];
            if port_mouth.port_flow_promotion.is_some() {
                return Err(ProductDesignError::InvalidReferenceEvidence(format!(
                    "product reference point {point_index} already contains port-flow promotion evidence"
                )));
            }

            let replayed = replay_port_flow_promotion(artifact, PortFlowPromotionGate::default())
                .map_err(|error| {
                ProductDesignError::InvalidReferenceEvidence(format!(
                    "invalid port-flow runtime promotion artifact at point {point_index}: {error}"
                ))
            })?;
            let frequency_scale = replayed
                .evidence
                .frequency_hz
                .max(point.frequency_hz)
                .max(1.0);
            if (replayed.evidence.frequency_hz - point.frequency_hz).abs()
                > 1.0e-9 * frequency_scale
            {
                return Err(ProductDesignError::InvalidReferenceEvidence(format!(
                    "port-flow runtime promotion frequency {} does not match product reference frequency {} at point {point_index}",
                    replayed.evidence.frequency_hz, point.frequency_hz,
                )));
            }
            replayed_evidence.push(replayed.evidence);
        }

        // The loop above proves every destination exists exactly once and is
        // empty, and every artifact has passed replay. This mutation phase has
        // no fallible validation path, preserving the all-or-nothing contract.
        for (point, evidence) in self.points.iter_mut().zip(replayed_evidence) {
            let interfaces = point
                .interface_evidence
                .as_mut()
                .expect("port-flow attachment destinations were prevalidated");
            let port_mouth = interfaces
                .iter_mut()
                .find(|interface| interface.interface_id == "port-mouth")
                .expect("port-mouth attachment destination was prevalidated");
            port_mouth.port_flow_promotion = Some(evidence);
        }

        Ok(())
    }

    pub fn maximum_relative_residual(&self) -> Option<f64> {
        self.points
            .iter()
            .map(|point| point.relative_residual)
            .reduce(f64::max)
    }

    /// Require the full-product evidence needed to promote a partitioned path
    /// against a monolithic reference.  This deliberately rejects legacy
    /// fixture sweeps instead of treating their absence as an implicit pass.
    pub fn validate_monolithic_product_evidence(
        &self,
        gate: MonolithicEvidenceGate,
    ) -> Result<(), ProductDesignError> {
        self.validate()?;
        gate.validate()?;
        if gate.require_mesh_convergence_ladder {
            if self.mesh_convergence.len() != self.points.len() {
                return Err(ProductDesignError::InvalidReferenceEvidence(
                    "strict product-scale promotion requires one mesh ladder per frequency".into(),
                ));
            }
            for point in &self.points {
                let ladder = self
                    .mesh_convergence
                    .iter()
                    .find(|ladder| ladder.frequency_hz == point.frequency_hz)
                    .ok_or_else(|| {
                        ProductDesignError::InvalidReferenceEvidence(
                            "strict product-scale promotion is missing a frequency mesh ladder"
                                .into(),
                        )
                    })?;
                let finest = ladder.finest_point().ok_or_else(|| {
                    ProductDesignError::InvalidReferenceEvidence(
                        "strict product-scale mesh ladder has no finest point".into(),
                    )
                })?;
                if retained_product_observable_error(point, finest)? > 1.0e-10 {
                    return Err(ProductDesignError::InvalidReferenceEvidence(
                        "reference sweep point differs from its retained finest mesh level".into(),
                    ));
                }
                let point_mesh = point.monolithic_evidence.as_ref().ok_or_else(|| {
                    ProductDesignError::InvalidReferenceEvidence(
                        "strict product-scale point is missing monolithic mesh identity".into(),
                    )
                })?;
                let finest_mesh = finest.monolithic_evidence.as_ref().ok_or_else(|| {
                    ProductDesignError::InvalidReferenceEvidence(
                        "strict product-scale finest level is missing monolithic mesh identity"
                            .into(),
                    )
                })?;
                if point_mesh.mesh_node_count != finest_mesh.mesh_node_count
                    || point_mesh.mesh_tetrahedron_count != finest_mesh.mesh_tetrahedron_count
                    || point_mesh.mesh_geometry_fingerprint != finest_mesh.mesh_geometry_fingerprint
                    || ladder.maximum_adjacent_observable_error > gate.max_mesh_convergence_error
                {
                    return Err(ProductDesignError::InvalidReferenceEvidence(
                        "strict product-scale mesh convergence exceeds promotion gate".into(),
                    ));
                }
            }
        }
        let mut expected_interface_ids = None;
        let mut expected_mesh_identity = None;
        for point in &self.points {
            if gate.require_structural_evidence {
                let structural = point.structural_evidence.as_ref().ok_or_else(|| {
                    ProductDesignError::InvalidReferenceEvidence(
                        "resolved product reference point missing structural evidence".into(),
                    )
                })?;
                structural.validate_for_point(
                    point.frequency_hz,
                    point.mean_pressure_rms_pa,
                    gate.structural,
                )?;
            }
            if gate.require_thermal_evidence {
                let thermal = point.thermal_evidence.as_ref().ok_or_else(|| {
                    ProductDesignError::InvalidReferenceEvidence(
                        "resolved product reference point missing thermal evidence".into(),
                    )
                })?;
                thermal.validate_for_driver_count(
                    point.driver_velocities_rms_m_per_s.len(),
                    gate.thermal,
                )?;
            }
            let interfaces = point.interface_evidence.as_ref().ok_or_else(|| {
                ProductDesignError::InvalidReferenceEvidence(
                    "resolved product reference point is missing interface evidence".into(),
                )
            })?;
            if interfaces.is_empty() {
                return Err(ProductDesignError::InvalidReferenceEvidence(
                    "resolved product reference point has no interfaces".into(),
                ));
            }
            let interface_ids = interfaces
                .iter()
                .map(|interface| interface.interface_id.as_str())
                .collect::<BTreeSet<_>>();
            if let Some(expected) = expected_interface_ids.as_ref() {
                if *expected != interface_ids {
                    return Err(ProductDesignError::InvalidReferenceEvidence(
                        "resolved product reference interface identities differ across frequencies"
                            .into(),
                    ));
                }
            } else {
                expected_interface_ids = Some(interface_ids);
            }
            for interface in interfaces {
                interface.validate()?;
                if gate.require_port_flow_promotion && interface.interface_id == "port-mouth" {
                    let evidence = interface.port_flow_promotion.as_ref().ok_or_else(|| {
                        ProductDesignError::InvalidReferenceEvidence(
                            "ported monolithic reference missing nonlinear port-flow promotion evidence"
                                .into(),
                        )
                    })?;
                    let frequency_scale = evidence.frequency_hz.max(point.frequency_hz).max(1.0);
                    if (evidence.frequency_hz - point.frequency_hz).abs() > 1.0e-9 * frequency_scale
                    {
                        return Err(ProductDesignError::InvalidReferenceEvidence(
                            "port-flow promotion frequency differs reference point".into(),
                        ));
                    }
                }
                let mesh_convergence_error = interface.mesh_convergence_error.ok_or_else(|| {
                    ProductDesignError::InvalidReferenceEvidence(
                        "resolved product reference interface is missing mesh-convergence evidence"
                            .into(),
                    )
                })?;
                if interface.relative_residual > gate.max_block_relative_residual
                    || interface.power_residual_w.abs() > gate.max_power_residual_w
                    || mesh_convergence_error > gate.max_mesh_convergence_error
                {
                    return Err(ProductDesignError::InvalidReferenceEvidence(
                        "interface product reference evidence exceeds promotion gate".into(),
                    ));
                }
            }
            let evidence = point.monolithic_evidence.as_ref().ok_or_else(|| {
                ProductDesignError::InvalidReferenceEvidence(
                    "resolved product reference point is missing monolithic evidence".into(),
                )
            })?;
            evidence.validate()?;
            if gate.require_independent_validation {
                let validation = evidence.independent_validation.as_ref().ok_or_else(|| {
                    ProductDesignError::InvalidReferenceEvidence(
                        "strict product promotion requires independent validation evidence".into(),
                    )
                })?;
                validation.validate_for_point(point, &gate)?;
            }
            let mesh_identity = (
                evidence.mesh_node_count,
                evidence.mesh_tetrahedron_count,
                evidence.mesh_geometry_fingerprint.as_str(),
            );
            if let Some(expected) = expected_mesh_identity {
                if expected != mesh_identity {
                    return Err(ProductDesignError::InvalidReferenceEvidence(
                        "resolved product reference mesh identity differs across frequencies"
                            .into(),
                    ));
                }
            } else {
                expected_mesh_identity = Some(mesh_identity);
            }
            let max_residual = evidence
                .interior_relative_residual
                .max(evidence.exterior_relative_residual)
                .max(evidence.mechanical_relative_residual)
                .max(evidence.electrical_relative_residual);
            if max_residual > gate.max_block_relative_residual
                || evidence.electrical_power_residual_w.abs() > gate.max_power_residual_w
                || evidence.mechanical_power_residual_w.abs() > gate.max_power_residual_w
                || evidence.maximum_port_pressure_jump_pa > gate.max_port_pressure_jump_pa
            {
                return Err(ProductDesignError::InvalidReferenceEvidence(
                    "monolithic product reference evidence exceeds promotion gate".into(),
                ));
            }
        }
        Ok(())
    }

    /// Promote this partitioned sweep only when it agrees with a resolved
    /// monolithic product reference.  Both artifacts must retain the full
    /// dynamic driver vector and the same named interfaces at every frequency;
    /// a legacy equivalent-cylinder result cannot be promoted by this API.
    ///
    /// The compared observables deliberately include complex driver velocity,
    /// port volume flow, and cavity pressure.  Checking only scalar SPL would
    /// otherwise admit a phase-inconsistent crossover or a cancelled port
    /// error that looks plausible at one microphone.
    pub fn validate_partitioned_product_promotion_against(
        &self,
        monolithic: &Self,
        monolithic_gate: MonolithicEvidenceGate,
        promotion_gate: PartitionedProductPromotionGate,
    ) -> Result<(), ProductDesignError> {
        self.validate()?;
        monolithic.validate_monolithic_product_evidence(monolithic_gate)?;
        promotion_gate.validate()?;

        if self.driver_count.max(1) != monolithic.driver_count.max(1) {
            return Err(ProductDesignError::InvalidReferenceEvidence(
                "partitioned and monolithic product driver counts differ".into(),
            ));
        }
        if self.points.len() != monolithic.points.len() {
            return Err(ProductDesignError::InvalidReferenceEvidence(
                "partitioned and monolithic product frequency grids differ".into(),
            ));
        }
        for (name, actual, reference) in [
            (
                "speed of sound",
                self.speed_of_sound_m_per_s,
                monolithic.speed_of_sound_m_per_s,
            ),
            (
                "density",
                self.density_kg_per_m3,
                monolithic.density_kg_per_m3,
            ),
        ] {
            if relative_scalar_error(actual, reference, promotion_gate.observable_absolute_floor)
                > promotion_gate.max_relative_observable_error
            {
                return Err(ProductDesignError::InvalidReferenceEvidence(format!(
                    "partitioned and monolithic product {name} differ"
                )));
            }
        }
        for (partitioned, reference) in self.points.iter().zip(&monolithic.points) {
            if relative_scalar_error(
                partitioned.frequency_hz,
                reference.frequency_hz,
                promotion_gate.observable_absolute_floor,
            ) > f64::EPSILON
            {
                return Err(ProductDesignError::InvalidReferenceEvidence(
                    "partitioned and monolithic product frequency grids differ".into(),
                ));
            }
            if partitioned.relative_residual > promotion_gate.max_partitioned_relative_residual {
                return Err(ProductDesignError::InvalidReferenceEvidence(
                    "partitioned product residual exceeds promotion gate".into(),
                ));
            }
            if partitioned.driver_velocities_rms_m_per_s.len()
                != reference.driver_velocities_rms_m_per_s.len()
            {
                return Err(ProductDesignError::InvalidReferenceEvidence(
                    "partitioned and monolithic driver velocity evidence differs".into(),
                ));
            }
            for (actual, expected) in partitioned
                .driver_velocities_rms_m_per_s
                .iter()
                .zip(&reference.driver_velocities_rms_m_per_s)
            {
                if relative_complex_error(
                    *actual,
                    *expected,
                    promotion_gate.observable_absolute_floor,
                ) > promotion_gate.max_relative_observable_error
                {
                    return Err(ProductDesignError::InvalidReferenceEvidence(
                        "partitioned driver velocity differs from monolithic reference".into(),
                    ));
                }
            }
            for (name, actual, expected) in [
                (
                    "drive velocity",
                    partitioned.drive_velocity_rms_m_per_s,
                    reference.drive_velocity_rms_m_per_s,
                ),
                (
                    "aperture volume velocity",
                    partitioned.aperture_volume_velocity_m3_per_s,
                    reference.aperture_volume_velocity_m3_per_s,
                ),
                (
                    "mean cavity pressure",
                    partitioned.mean_pressure_rms_pa,
                    reference.mean_pressure_rms_pa,
                ),
            ] {
                if relative_complex_error(
                    actual,
                    expected,
                    promotion_gate.observable_absolute_floor,
                ) > promotion_gate.max_relative_observable_error
                {
                    return Err(ProductDesignError::InvalidReferenceEvidence(format!(
                        "partitioned {name} differs from monolithic reference"
                    )));
                }
            }
            if relative_scalar_error(
                partitioned.mean_abs_pressure_pa,
                reference.mean_abs_pressure_pa,
                promotion_gate.observable_absolute_floor,
            ) > promotion_gate.max_relative_observable_error
            {
                return Err(ProductDesignError::InvalidReferenceEvidence(
                    "partitioned mean pressure magnitude differs from monolithic reference".into(),
                ));
            }

            let partitioned_interfaces =
                partitioned.interface_evidence.as_ref().ok_or_else(|| {
                    ProductDesignError::InvalidReferenceEvidence(
                        "partitioned product reference point is missing interface evidence".into(),
                    )
                })?;
            let reference_interfaces = reference.interface_evidence.as_ref().ok_or_else(|| {
                ProductDesignError::InvalidReferenceEvidence(
                    "monolithic product reference point is missing interface evidence".into(),
                )
            })?;
            let partitioned_ids = partitioned_interfaces
                .iter()
                .map(|interface| interface.interface_id.as_str())
                .collect::<BTreeSet<_>>();
            let reference_ids = reference_interfaces
                .iter()
                .map(|interface| interface.interface_id.as_str())
                .collect::<BTreeSet<_>>();
            if partitioned_ids != reference_ids {
                return Err(ProductDesignError::InvalidReferenceEvidence(
                    "partitioned and monolithic product interface identities differ".into(),
                ));
            }
            for interface in partitioned_interfaces {
                interface.validate()?;
                let mesh_convergence_error = interface.mesh_convergence_error.ok_or_else(|| {
                    ProductDesignError::InvalidReferenceEvidence(
                        "partitioned product interface is missing mesh-convergence evidence".into(),
                    )
                })?;
                if interface.relative_residual > promotion_gate.max_partitioned_relative_residual
                    || interface.power_residual_w.abs()
                        > promotion_gate.max_partitioned_power_residual_w
                    || mesh_convergence_error
                        > promotion_gate.max_partitioned_mesh_convergence_error
                {
                    return Err(ProductDesignError::InvalidReferenceEvidence(
                        "partitioned product interface evidence exceeds promotion gate".into(),
                    ));
                }
                let reference_interface = reference_interfaces
                    .iter()
                    .find(|candidate| candidate.interface_id == interface.interface_id)
                    .ok_or_else(|| {
                        ProductDesignError::InvalidReferenceEvidence(
                            "monolithic product interface is missing from comparison".into(),
                        )
                    })?;
                if relative_complex_error(
                    interface.volume_flow_m3_per_s,
                    reference_interface.volume_flow_m3_per_s,
                    promotion_gate.observable_absolute_floor,
                ) > promotion_gate.max_relative_observable_error
                {
                    return Err(ProductDesignError::InvalidReferenceEvidence(
                        "partitioned interface volume flow differs from monolithic reference"
                            .into(),
                    ));
                }
            }
        }
        Ok(())
    }
}

/// Versioned output artifact binding a resolved product reference sweep to the
/// exact validated product input that produced it.
pub const PRODUCT_REFERENCE_ARTIFACT_SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProductReferenceArtifact {
    pub schema_version: u32,
    pub provenance: ArtifactProvenance,
    pub product_content_sha256: String,
    pub sweep: ProductReferenceSweep,
}

impl ProductReferenceArtifact {
    /// Build a replayable product-result artifact with input identity binding.
    pub fn from_design(
        design: &ProductDesign,
        sweep: ProductReferenceSweep,
    ) -> Result<Self, ProductDesignError> {
        let input = design.to_artifact()?;
        let artifact = Self {
            schema_version: PRODUCT_REFERENCE_ARTIFACT_SCHEMA_VERSION,
            provenance: sweep.provenance,
            product_content_sha256: input.content_sha256,
            sweep,
        };
        artifact.validate_against_design(design)?;
        Ok(artifact)
    }

    /// Validate envelope, digest, provenance, and distributed sweep payload.
    pub fn validate(&self) -> Result<(), ProductDesignError> {
        if self.schema_version != PRODUCT_REFERENCE_ARTIFACT_SCHEMA_VERSION {
            return Err(ProductDesignError::InvalidProductArtifact(format!(
                "unsupported product reference artifact schema {}; expected {}",
                self.schema_version, PRODUCT_REFERENCE_ARTIFACT_SCHEMA_VERSION
            )));
        }
        if self.provenance.schema_version != super::conventions::ARTIFACT_PROVENANCE_SCHEMA_VERSION
        {
            return Err(ProductDesignError::InvalidProductArtifact(
                "unsupported product reference provenance schema".into(),
            ));
        }
        if !self.provenance.deterministic {
            return Err(ProductDesignError::InvalidProductArtifact(
                "product reference artifact provenance must be deterministic".into(),
            ));
        }
        if self.product_content_sha256.len() != 64
            || !self
                .product_content_sha256
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        {
            return Err(ProductDesignError::InvalidProductArtifact(
                "product reference input digest must be lowercase SHA-256".into(),
            ));
        }
        self.sweep.validate()
    }

    /// Revalidate that the result belongs to the supplied product input.
    pub fn validate_against_design(
        &self,
        design: &ProductDesign,
    ) -> Result<(), ProductDesignError> {
        self.validate()?;
        let input = design.to_artifact()?;
        if self.product_content_sha256 != input.content_sha256 {
            return Err(ProductDesignError::InvalidProductArtifact(
                "product reference input digest mismatch".into(),
            ));
        }
        let expected_driver_count = design.drivers.len();
        if self.sweep.driver_count.max(1) != expected_driver_count {
            return Err(ProductDesignError::InvalidReferenceEvidence(
                "product reference driver count differs bound design".into(),
            ));
        }
        if self.sweep.points.iter().any(|point| {
            !point.driver_velocities_rms_m_per_s.is_empty()
                && point.driver_velocities_rms_m_per_s.len() != expected_driver_count
        }) {
            return Err(ProductDesignError::InvalidReferenceEvidence(
                "product reference driver evidence differs bound design".into(),
            ));
        }
        Ok(())
    }

    /// Serialize deterministic field ordering and bounded float precision.
    pub fn canonical_json(&self) -> Result<Vec<u8>, serde_json::Error> {
        let mut value = serde_json::to_value(self)?;
        canonicalize_product_json(&mut value);
        serde_json::to_vec(&value)
    }

    /// Deserialize and validate before exposing the retained sweep.
    pub fn from_json(bytes: &[u8]) -> Result<Self, ProductDesignError> {
        let artifact: Self = serde_json::from_slice(bytes)
            .map_err(|error| ProductDesignError::InvalidProductArtifact(error.to_string()))?;
        artifact.validate()?;
        Ok(artifact)
    }

    /// Consume a validated result artifact.
    pub fn into_sweep(self) -> Result<ProductReferenceSweep, ProductDesignError> {
        self.validate()?;
        Ok(self.sweep)
    }
}

impl ProductDesign {
    pub fn validate(&self) -> Result<(), ProductDesignError> {
        self.cabinet.validate()?;
        if self.drivers.is_empty() {
            return Err(ProductDesignError::NoDrivers);
        }
        self.crossover.validate(self.drivers.len())?;
        if let Some(waveguide) = &self.waveguide {
            let Some((driver_index, driver)) = self
                .drivers
                .iter()
                .enumerate()
                .find(|(_, driver)| driver.id == waveguide.driver_id)
            else {
                return Err(ProductDesignError::InvalidWaveguide(format!(
                    "waveguide target driver {:?} does not exist",
                    waveguide.driver_id
                )));
            };
            waveguide
                .validate_for_aperture(driver.radius_m)
                .map_err(ProductDesignError::InvalidWaveguide)?;

            let horizontal_rim_m = waveguide.horizontal_spline[0][1];
            let vertical_rim_m = waveguide.vertical_spline[0][1];
            if driver.position_m[0].abs() + horizontal_rim_m > self.cabinet.baffle_width_m / 2.0
                || driver.position_m[1].abs() + vertical_rim_m > self.cabinet.baffle_height_m / 2.0
            {
                return Err(ProductDesignError::InvalidWaveguide(format!(
                    "waveguide target driver {driver_index} rim lies outside baffle"
                )));
            }
        }
        if let Some(structural) = &self.structural_acoustic {
            structural.validate().map_err(|error| {
                ProductDesignError::InvalidStructuralFeedback(error.to_string())
            })?;
        }
        let mut driver_ids = BTreeSet::new();
        let mut crossover_branch_owners = vec![None; self.crossover.branches.len()];
        for (index, driver) in self.drivers.iter().enumerate() {
            driver.validate(
                index,
                self.cabinet.baffle_width_m,
                self.cabinet.baffle_height_m,
                self.cabinet.box_width_m,
                self.cabinet.box_height_m,
                self.cabinet.box_depth_m,
            )?;
            if !driver_ids.insert(driver.id.as_str()) {
                return Err(ProductDesignError::DuplicateDriverId {
                    id: driver.id.clone(),
                });
            }
            if driver.crossover_branch >= self.crossover.branches.len() {
                return Err(ProductDesignError::InvalidDriver {
                    index,
                    reason: "crossover branch index is out of range",
                });
            }
            if let Some(first_driver) = crossover_branch_owners[driver.crossover_branch] {
                return Err(ProductDesignError::CrossoverBranchReuse {
                    branch: driver.crossover_branch,
                    first_driver,
                    second_driver: index,
                });
            }
            crossover_branch_owners[driver.crossover_branch] = Some(index);
        }
        for (left_index, left) in self.drivers.iter().enumerate() {
            for (right_index, right) in self.drivers.iter().enumerate().skip(left_index + 1) {
                let dx = left.position_m[0] - right.position_m[0];
                let dy = left.position_m[1] - right.position_m[1];
                let dz = left.position_m[2] - right.position_m[2];
                let separation = (dx * dx + dy * dy + dz * dz).sqrt();
                if separation <= left.radius_m + right.radius_m + 1.0e-3 {
                    return Err(ProductDesignError::DriversOverlap {
                        left: left_index,
                        right: right_index,
                    });
                }
            }
        }
        if let Some(port) = &self.cabinet.port {
            for (index, driver) in self.drivers.iter().enumerate() {
                if driver.normal[2] < 1.0 - 1.0e-9 {
                    continue;
                }
                let dx = driver.position_m[0] - port.position_m[0];
                let dy = driver.position_m[1] - port.position_m[1];
                let separation = (dx * dx + dy * dy).sqrt();
                // A resolved product mesh contains the tube as a solid.  Its
                // clearance must therefore use the exterior wall radius, not
                // merely the air-bore radius retained by legacy lumped ports.
                let port_clearance_radius_m =
                    port.outer_diameter_m().unwrap_or(port.diameter_m) * 0.5;
                if separation <= driver.radius_m + port_clearance_radius_m + 1.0e-3 {
                    return Err(ProductDesignError::PortOverlapsDriver { index });
                }
            }
        }
        Ok(())
    }

    /// Volume-velocity impedance seen at the exterior mouth of the port.
    ///
    /// The port and cabinet compliance are in series for port-mouth volume
    /// flow. The lumped port model supplies the existing end correction and
    /// radiation resistance, while exterior BEM supplies the resolved
    /// aperture field. Full distributed interior/exterior feedback remains a
    /// higher-fidelity model.
    ///
    /// The result uses the modeling crate's `e^{+iωt}` convention. Consumers
    /// using the speaker BEM convention must conjugate it before forming a
    /// boundary admittance.
    pub fn port_boundary_impedance_at(
        &self,
        frequency_hz: f64,
        medium: AcousticMedium,
    ) -> Result<Option<Complex64>, ProductDesignError> {
        self.validate()?;
        if !frequency_hz.is_finite() || frequency_hz <= 0.0 {
            return Err(ProductDesignError::InvalidFrequency);
        }
        let Some(port) = &self.cabinet.port else {
            return Ok(None);
        };
        medium
            .validate()
            .map_err(|error| ProductDesignError::InvalidPortBoundary(error.to_string()))?;
        let enclosure = self.cabinet.to_lumped()?;
        let EnclosureSpec::Vented { volume_m3, .. } = enclosure else {
            return Ok(None);
        };
        let port_spec = super::PortSpec {
            radius_m: port.diameter_m / 2.0,
            length_m: port.length_m,
            end_correction_m: port.end_correction_m(),
            resistance_pa_s_per_m3: 0.0,
        };
        let port_impedance = port_spec
            .acoustic_impedance(frequency_hz, medium)
            .map_err(|error| ProductDesignError::InvalidPortBoundary(error.to_string()))?;
        let omega = std::f64::consts::TAU * frequency_hz;
        let compliance =
            volume_m3 / (medium.density_kg_per_m3 * medium.sound_speed_m_per_s.powi(2));
        let cavity_impedance = Complex64::new(0.0, -1.0 / (omega * compliance));
        let impedance = port_impedance + cavity_impedance;
        if !impedance.re.is_finite() || !impedance.im.is_finite() || impedance.re < -1.0e-12 {
            return Err(ProductDesignError::InvalidPortBoundary(
                "port boundary impedance must be finite and passive".into(),
            ));
        }
        Ok(Some(impedance))
    }

    /// Build the product-owned body-fitted monolithic FEM–BEM reference.
    ///
    /// This is the product-facing entry point for the resolved geometry
    /// path. It retains the canonical product mesh and patch registry in the
    /// returned coupling, so callers do not silently fall back to the
    /// equivalent-cylinder reference used by the legacy sweep API.
    pub fn build_body_fitted_monolithic_reference(
        &self,
        mesh_config: ProductInteriorMeshConfig,
        speed_of_sound: f64,
        density: f64,
    ) -> Result<MonolithicSealedProductCoupling, ProductDesignError> {
        self.validate()?;
        if !speed_of_sound.is_finite()
            || speed_of_sound <= 0.0
            || !density.is_finite()
            || density <= 0.0
        {
            return Err(ProductDesignError::InvalidReferenceEvidence(
                "body-fitted reference medium must have finite positive speed and density".into(),
            ));
        }
        let mesh = build_body_fitted_sealed_product_interior_mesh(self, mesh_config)
            .map_err(|error| ProductDesignError::ReferenceCoupling(error.to_string()))?;
        MonolithicSealedProductCoupling::new(self, mesh, speed_of_sound, density)
            .map_err(|error| ProductDesignError::ReferenceCoupling(error.to_string()))
    }

    /// Solve the product-owned body-fitted FEM–BEM reference on a frequency
    /// grid and retain validated full-order product observables.
    ///
    /// This adapter deliberately retains only the one mesh actually solved:
    /// `mesh_convergence` remains empty until independently refined product
    /// meshes have been solved and compared. Likewise, no ROM promotion
    /// evidence is inferred from the full-order solve.
    pub fn solve_body_fitted_monolithic_reference_sweep(
        &self,
        mesh_config: ProductInteriorMeshConfig,
        speed_of_sound: f64,
        density: f64,
        frequencies_hz: &[f64],
        input_voltage_rms: Complex64,
    ) -> Result<ProductReferenceSweep, ProductDesignError> {
        if frequencies_hz.is_empty() {
            return Err(ProductDesignError::ReferenceSweepEmpty);
        }
        if !input_voltage_rms.is_finite() {
            return Err(ProductDesignError::InvalidVoltage);
        }
        if frequencies_hz
            .iter()
            .any(|frequency_hz| !frequency_hz.is_finite() || *frequency_hz <= 0.0)
        {
            return Err(ProductDesignError::InvalidFrequency);
        }

        let coupling =
            self.build_body_fitted_monolithic_reference(mesh_config, speed_of_sound, density)?;
        let mesh = coupling.interior_mesh();
        let aperture_elements = mesh
            .patches
            .iter()
            .filter(|patch| {
                matches!(
                    patch.kind,
                    ProductInteriorPatchKind::DriverRear { .. }
                        | ProductInteriorPatchKind::PortMouth
                )
            })
            .map(|patch| patch.boundary_faces.len())
            .sum();
        let mut points = Vec::with_capacity(frequencies_hz.len());

        for &frequency_hz in frequencies_hz {
            let solution = coupling
                .solve_frequency(self, frequency_hz, input_voltage_rms)
                .map_err(|error| ProductDesignError::ReferenceCoupling(error.to_string()))?;
            points.push(
                ProductReferencePoint::from_monolithic_solution_with_structural_feedback(
                    &solution,
                    mesh,
                    0.0,
                    self.structural_acoustic.as_ref(),
                )?,
            );
        }

        let sweep = ProductReferenceSweep {
            speed_of_sound_m_per_s: speed_of_sound,
            density_kg_per_m3: density,
            interior_dofs: mesh.diagnostics.node_count,
            aperture_elements,
            points,
            driver_count: self.drivers.len(),
            provenance: ArtifactProvenance::native_golden(),
            rom_promotion: None,
            mesh_convergence: Vec::new(),
        };
        sweep.validate_monolithic_product_evidence(MonolithicEvidenceGate::default())?;
        Ok(sweep)
    }

    fn validate_body_fitted_thermal_inputs(
        &self,
        frequencies_hz: &[f64],
        thermal_config: DriverThermalNetworkConfig,
        eddy_loss_w: &[Vec<f64>],
    ) -> Result<(), ProductDesignError> {
        self.validate()?;
        thermal_config.validate().map_err(|error| {
            ProductDesignError::InvalidReferenceEvidence(format!(
                "invalid body-fitted thermal configuration: {error}"
            ))
        })?;
        if eddy_loss_w.len() != frequencies_hz.len() {
            return Err(ProductDesignError::InvalidReferenceEvidence(
                "body-fitted thermal loss grid must contain one row per frequency".into(),
            ));
        }
        for (frequency_index, losses) in eddy_loss_w.iter().enumerate() {
            if losses.len() != self.drivers.len() {
                return Err(ProductDesignError::InvalidReferenceEvidence(format!(
                    "body-fitted thermal loss row {frequency_index} driver count differs from product"
                )));
            }
            if losses.iter().any(|loss| !loss.is_finite() || *loss < 0.0) {
                return Err(ProductDesignError::InvalidReferenceEvidence(format!(
                    "body-fitted thermal loss row {frequency_index} contains a non-finite or negative loss"
                )));
            }
        }
        Ok(())
    }

    /// Solve the product-owned body-fitted FEM–BEM reference while retaining
    /// thermal-network evidence from the same electromechanical state.
    ///
    /// The monolithic solve supplies copper loss through its solved currents;
    /// `eddy_loss_w` supplies the separately identified eddy-current channel
    /// for each frequency and driver. The loss grid is intentionally explicit
    /// so this adapter does not claim that the acoustic solve itself computed
    /// eddy loss or closed a temperature-dependent feedback iteration.
    pub fn solve_body_fitted_monolithic_reference_sweep_with_thermal(
        &self,
        mesh_config: ProductInteriorMeshConfig,
        speed_of_sound: f64,
        density: f64,
        frequencies_hz: &[f64],
        input_voltage_rms: Complex64,
        thermal_config: DriverThermalNetworkConfig,
        eddy_loss_w: &[Vec<f64>],
    ) -> Result<ProductReferenceSweep, ProductDesignError> {
        if frequencies_hz.is_empty() {
            return Err(ProductDesignError::ReferenceSweepEmpty);
        }
        if !input_voltage_rms.is_finite() {
            return Err(ProductDesignError::InvalidVoltage);
        }
        if frequencies_hz
            .iter()
            .any(|frequency_hz| !frequency_hz.is_finite() || *frequency_hz <= 0.0)
        {
            return Err(ProductDesignError::InvalidFrequency);
        }
        self.validate_body_fitted_thermal_inputs(frequencies_hz, thermal_config, eddy_loss_w)?;

        let coupling =
            self.build_body_fitted_monolithic_reference(mesh_config, speed_of_sound, density)?;
        let mesh = coupling.interior_mesh();
        let aperture_elements = mesh
            .patches
            .iter()
            .filter(|patch| {
                matches!(
                    patch.kind,
                    ProductInteriorPatchKind::DriverRear { .. }
                        | ProductInteriorPatchKind::PortMouth
                )
            })
            .map(|patch| patch.boundary_faces.len())
            .sum();
        let mut points = Vec::with_capacity(frequencies_hz.len());

        for (frequency_index, &frequency_hz) in frequencies_hz.iter().enumerate() {
            let solution = coupling
                .solve_frequency(self, frequency_hz, input_voltage_rms)
                .map_err(|error| ProductDesignError::ReferenceCoupling(error.to_string()))?;
            points.push(self.reference_point_from_monolithic_solution_with_thermal(
                &solution,
                mesh,
                0.0,
                thermal_config,
                &eddy_loss_w[frequency_index],
            )?);
        }

        let sweep = ProductReferenceSweep {
            speed_of_sound_m_per_s: speed_of_sound,
            density_kg_per_m3: density,
            interior_dofs: mesh.diagnostics.node_count,
            aperture_elements,
            points,
            driver_count: self.drivers.len(),
            provenance: ArtifactProvenance::native_golden(),
            rom_promotion: None,
            mesh_convergence: Vec::new(),
        };
        sweep.validate_monolithic_product_evidence(MonolithicEvidenceGate::default())?;
        Ok(sweep)
    }

    /// Execute independently meshed body-fitted monolithic references and
    /// retain a per-frequency product mesh-convergence ladder.
    ///
    /// The returned ladder contains actual solved states, not synthetic
    /// perturbations. It is validated structurally and with the default
    /// residual/power gate, but callers must still opt into
    /// `MonolithicEvidenceGate::strict_product_scale` before promotion.
    /// Solve a body-fitted reference sweep with temperature-dependent copper
    /// resistance fed back into the canonical electromechanical block.
    ///
    /// The supplied eddy-loss grid remains an explicit calibrated source;
    /// this method closes the copper-resistance/thermal-state fixed point and
    /// retains the converged hot resistances in the product evidence.
    #[expect(
        clippy::too_many_arguments,
        reason = "monolithic reference-sweep entry point threading mesh, thermal state and evidence sinks; grouping would churn the sweep call sites"
    )]
    pub fn solve_body_fitted_monolithic_reference_sweep_with_thermal_feedback(
        &self,
        mesh_config: ProductInteriorMeshConfig,
        speed_of_sound: f64,
        density: f64,
        frequencies_hz: &[f64],
        input_voltage_rms: Complex64,
        thermal_config: DriverThermalNetworkConfig,
        eddy_loss_w: &[Vec<f64>],
        max_iterations: usize,
        resistance_tolerance: f64,
    ) -> Result<ProductReferenceSweep, ProductDesignError> {
        if max_iterations == 0 || !resistance_tolerance.is_finite() || resistance_tolerance <= 0.0 {
            return Err(ProductDesignError::InvalidReferenceEvidence(
                "thermal feedback iteration budget and resistance tolerance must be positive"
                    .into(),
            ));
        }
        if frequencies_hz.is_empty() {
            return Err(ProductDesignError::ReferenceSweepEmpty);
        }
        if !input_voltage_rms.is_finite() {
            return Err(ProductDesignError::InvalidVoltage);
        }
        if frequencies_hz
            .iter()
            .any(|frequency_hz| !frequency_hz.is_finite() || *frequency_hz <= 0.0)
        {
            return Err(ProductDesignError::InvalidFrequency);
        }
        self.validate_body_fitted_thermal_inputs(frequencies_hz, thermal_config, eddy_loss_w)?;
        let coupling =
            self.build_body_fitted_monolithic_reference(mesh_config, speed_of_sound, density)?;
        let mesh = coupling.interior_mesh();
        let aperture_elements = mesh
            .patches
            .iter()
            .filter(|patch| {
                matches!(
                    patch.kind,
                    ProductInteriorPatchKind::DriverRear { .. }
                        | ProductInteriorPatchKind::PortMouth
                )
            })
            .map(|patch| patch.boundary_faces.len())
            .sum();
        let mut points = Vec::with_capacity(frequencies_hz.len());
        for (frequency_index, &frequency_hz) in frequencies_hz.iter().enumerate() {
            let mut coil_resistances = self
                .drivers
                .iter()
                .map(|driver| driver.coil_resistance_ohm)
                .collect::<Vec<_>>();
            let mut final_change = f64::INFINITY;
            let mut converged = false;
            let mut iterations = 0;
            for iteration in 1..=max_iterations {
                let solution = coupling
                    .solve_frequency_with_coil_resistances(
                        self,
                        frequency_hz,
                        input_voltage_rms,
                        Some(&coil_resistances),
                    )
                    .map_err(|error| ProductDesignError::ReferenceCoupling(error.to_string()))?;
                let mut updated_resistances = Vec::with_capacity(self.drivers.len());
                for (((driver, current), &eddy_loss), &old_resistance) in self
                    .drivers
                    .iter()
                    .zip(&solution.driver_current_a)
                    .zip(&eddy_loss_w[frequency_index])
                    .zip(&coil_resistances)
                {
                    let power = thermal_config
                        .power_from_losses(old_resistance * current.norm_sqr(), eddy_loss)
                        .map_err(|error| {
                            ProductDesignError::InvalidReferenceEvidence(format!(
                                "invalid thermal feedback source: {error}"
                            ))
                        })?;
                    let state = thermal_config
                        .specification
                        .solve_steady_state(power)
                        .map_err(|error| {
                            ProductDesignError::InvalidReferenceEvidence(format!(
                                "thermal feedback solve failed: {error}"
                            ))
                        })?;
                    let target_resistance = thermal_config
                        .copper_resistance_at(
                            driver.coil_resistance_ohm,
                            state.temperatures_k[ThermalNode::VoiceCoil as usize],
                        )
                        .map_err(|error| {
                            ProductDesignError::InvalidReferenceEvidence(format!(
                                "invalid thermal feedback copper resistance: {error}"
                            ))
                        })?;
                    updated_resistances.push(0.5 * (old_resistance + target_resistance));
                }
                final_change = updated_resistances
                    .iter()
                    .zip(&coil_resistances)
                    .map(|(updated, old)| {
                        (updated - old).abs() / updated.abs().max(old.abs()).max(1.0e-12)
                    })
                    .fold(0.0, f64::max);
                coil_resistances = updated_resistances;
                iterations = iteration;
                if final_change <= resistance_tolerance {
                    converged = true;
                    break;
                }
            }
            if !converged {
                return Err(ProductDesignError::InvalidReferenceEvidence(format!(
                    "thermal feedback did not converge at frequency {frequency_hz} Hz after {iterations} iterations (relative resistance change {final_change})"
                )));
            }
            let solution = coupling
                .solve_frequency_with_coil_resistances(
                    self,
                    frequency_hz,
                    input_voltage_rms,
                    Some(&coil_resistances),
                )
                .map_err(|error| ProductDesignError::ReferenceCoupling(error.to_string()))?;
            let mut point =
                ProductReferencePoint::from_monolithic_solution_with_structural_feedback(
                    &solution,
                    mesh,
                    0.0,
                    self.structural_acoustic.as_ref(),
                )?;
            point.thermal_evidence = Some(self.thermal_reference_evidence_with_feedback(
                &solution,
                thermal_config,
                &eddy_loss_w[frequency_index],
                &coil_resistances,
                iterations,
                final_change,
            )?);
            point.validate()?;
            points.push(point);
        }
        let sweep = ProductReferenceSweep {
            speed_of_sound_m_per_s: speed_of_sound,
            density_kg_per_m3: density,
            interior_dofs: mesh.diagnostics.node_count,
            aperture_elements,
            points,
            driver_count: self.drivers.len(),
            provenance: ArtifactProvenance::native_golden(),
            rom_promotion: None,
            mesh_convergence: Vec::new(),
        };
        sweep.validate_monolithic_product_evidence(MonolithicEvidenceGate::default())?;
        Ok(sweep)
    }

    pub fn solve_body_fitted_monolithic_mesh_convergence_sweep(
        &self,
        mesh_levels: &[ProductBodyFittedMeshLevel],
        speed_of_sound: f64,
        density: f64,
        frequencies_hz: &[f64],
        input_voltage_rms: Complex64,
    ) -> Result<ProductReferenceSweep, ProductDesignError> {
        if mesh_levels.len() < 3 {
            return Err(ProductDesignError::InvalidReferenceEvidence(
                "body-fitted mesh convergence requires at least three levels".into(),
            ));
        }
        if frequencies_hz.is_empty() {
            return Err(ProductDesignError::ReferenceSweepEmpty);
        }
        if !input_voltage_rms.is_finite() {
            return Err(ProductDesignError::InvalidVoltage);
        }
        if frequencies_hz
            .iter()
            .any(|frequency_hz| !frequency_hz.is_finite() || *frequency_hz <= 0.0)
        {
            return Err(ProductDesignError::InvalidFrequency);
        }
        if frequencies_hz
            .windows(2)
            .any(|window| window[1] <= window[0])
        {
            return Err(ProductDesignError::InvalidReferenceEvidence(
                "body-fitted mesh convergence frequencies must be strictly increasing".into(),
            ));
        }

        let mut labels = BTreeSet::new();
        let mut previous_size_m = None;
        for (index, level) in mesh_levels.iter().enumerate() {
            let size_m = level.mesh_config.max_element_size_m;
            if level.label.trim().is_empty()
                || !labels.insert(level.label.as_str())
                || !size_m.is_finite()
                || size_m <= 0.0
                || level.mesh_config.min_cells_per_axis == 0
                || level.mesh_config.max_cells_per_axis < level.mesh_config.min_cells_per_axis
                || previous_size_m.is_some_and(|previous| size_m >= previous)
            {
                return Err(ProductDesignError::InvalidReferenceEvidence(format!(
                    "invalid or non-refining body-fitted mesh level {index}"
                )));
            }
            previous_size_m = Some(size_m);
        }

        let level_sweeps = mesh_levels
            .iter()
            .map(|level| {
                self.solve_body_fitted_monolithic_reference_sweep(
                    level.mesh_config,
                    speed_of_sound,
                    density,
                    frequencies_hz,
                    input_voltage_rms,
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        let finest = level_sweeps.last().ok_or_else(|| {
            ProductDesignError::InvalidReferenceEvidence(
                "body-fitted mesh convergence has no finest sweep".into(),
            )
        })?;
        let mut mesh_convergence = Vec::with_capacity(frequencies_hz.len());
        for frequency_index in 0..frequencies_hz.len() {
            let levels = mesh_levels
                .iter()
                .zip(&level_sweeps)
                .map(|(level, sweep)| ProductMeshConvergenceLevel {
                    label: level.label.clone(),
                    characteristic_size_m: level.mesh_config.max_element_size_m,
                    point: sweep.points[frequency_index].clone(),
                })
                .collect();
            mesh_convergence.push(ProductMeshConvergenceEvidence::from_levels(levels)?);
        }

        let sweep = ProductReferenceSweep {
            speed_of_sound_m_per_s: finest.speed_of_sound_m_per_s,
            density_kg_per_m3: finest.density_kg_per_m3,
            interior_dofs: finest.interior_dofs,
            aperture_elements: finest.aperture_elements,
            points: finest.points.clone(),
            driver_count: finest.driver_count,
            provenance: ArtifactProvenance::native_golden(),
            rom_promotion: None,
            mesh_convergence,
        };
        sweep.validate_monolithic_product_evidence(MonolithicEvidenceGate::default())?;
        Ok(sweep)
    }

    /// Execute independently meshed body-fitted monolithic references while
    /// retaining thermal evidence at every frequency and mesh level.
    ///
    /// The returned finest-level points and all retained ladder points use the
    /// same validated per-frequency/per-driver loss grid. Promotion remains
    /// caller-gated through a `MonolithicEvidenceGate` that requests both
    /// mesh-ladder and thermal evidence.
    pub fn solve_body_fitted_monolithic_mesh_convergence_sweep_with_thermal(
        &self,
        mesh_levels: &[ProductBodyFittedMeshLevel],
        speed_of_sound: f64,
        density: f64,
        frequencies_hz: &[f64],
        input_voltage_rms: Complex64,
        thermal_config: DriverThermalNetworkConfig,
        eddy_loss_w: &[Vec<f64>],
    ) -> Result<ProductReferenceSweep, ProductDesignError> {
        if mesh_levels.len() < 3 {
            return Err(ProductDesignError::InvalidReferenceEvidence(
                "body-fitted mesh convergence requires at least three levels".into(),
            ));
        }
        if frequencies_hz.is_empty() {
            return Err(ProductDesignError::ReferenceSweepEmpty);
        }
        if !input_voltage_rms.is_finite() {
            return Err(ProductDesignError::InvalidVoltage);
        }
        if frequencies_hz
            .iter()
            .any(|frequency_hz| !frequency_hz.is_finite() || *frequency_hz <= 0.0)
        {
            return Err(ProductDesignError::InvalidFrequency);
        }
        if frequencies_hz
            .windows(2)
            .any(|window| window[1] <= window[0])
        {
            return Err(ProductDesignError::InvalidReferenceEvidence(
                "body-fitted mesh convergence frequencies must be strictly increasing".into(),
            ));
        }
        self.validate_body_fitted_thermal_inputs(frequencies_hz, thermal_config, eddy_loss_w)?;

        let mut labels = BTreeSet::new();
        let mut previous_size_m = None;
        for (index, level) in mesh_levels.iter().enumerate() {
            let size_m = level.mesh_config.max_element_size_m;
            if level.label.trim().is_empty()
                || !labels.insert(level.label.as_str())
                || !size_m.is_finite()
                || size_m <= 0.0
                || level.mesh_config.min_cells_per_axis == 0
                || level.mesh_config.max_cells_per_axis < level.mesh_config.min_cells_per_axis
                || previous_size_m.is_some_and(|previous| size_m >= previous)
            {
                return Err(ProductDesignError::InvalidReferenceEvidence(format!(
                    "invalid or non-refining body-fitted mesh level {index}"
                )));
            }
            previous_size_m = Some(size_m);
        }

        let level_sweeps = mesh_levels
            .iter()
            .map(|level| {
                self.solve_body_fitted_monolithic_reference_sweep_with_thermal(
                    level.mesh_config,
                    speed_of_sound,
                    density,
                    frequencies_hz,
                    input_voltage_rms,
                    thermal_config,
                    eddy_loss_w,
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        let finest = level_sweeps.last().ok_or_else(|| {
            ProductDesignError::InvalidReferenceEvidence(
                "body-fitted mesh convergence has no finest sweep".into(),
            )
        })?;
        let mut mesh_convergence = Vec::with_capacity(frequencies_hz.len());
        for frequency_index in 0..frequencies_hz.len() {
            let levels = mesh_levels
                .iter()
                .zip(&level_sweeps)
                .map(|(level, sweep)| ProductMeshConvergenceLevel {
                    label: level.label.clone(),
                    characteristic_size_m: level.mesh_config.max_element_size_m,
                    point: sweep.points[frequency_index].clone(),
                })
                .collect();
            mesh_convergence.push(ProductMeshConvergenceEvidence::from_levels(levels)?);
        }

        let sweep = ProductReferenceSweep {
            speed_of_sound_m_per_s: finest.speed_of_sound_m_per_s,
            density_kg_per_m3: finest.density_kg_per_m3,
            interior_dofs: finest.interior_dofs,
            aperture_elements: finest.aperture_elements,
            points: finest.points.clone(),
            driver_count: finest.driver_count,
            provenance: ArtifactProvenance::native_golden(),
            rom_promotion: None,
            mesh_convergence,
        };
        sweep.validate_monolithic_product_evidence(MonolithicEvidenceGate::default())?;
        Ok(sweep)
    }

    /// Build the distributed FEM–BEM reference model for a product.
    ///
    /// The current fixture is an equivalent axisymmetric cylinder.  The
    /// interior driver boundary is nevertheless assembled from localized
    /// spatial profiles, so multiple non-overlapping planar drivers retain
    /// distributed mutual cavity loading.  The coaxial port limitation is
    /// explicit because this fixture has one axisymmetric exterior aperture.
    pub fn build_fem_bem_reference(
        &self,
        discretization: CylinderFixtureDiscretization,
        speed_of_sound: f64,
        density: f64,
    ) -> Result<FemBemCoupling, ProductDesignError> {
        self.validate()?;
        if let Some(port) = &self.cabinet.port
            && (port.position_m[0].abs() > 1.0e-12 || port.position_m[1].abs() > 1.0e-12)
        {
            return Err(ProductDesignError::ReferenceModelRequiresCentredPort);
        }
        let driver_aperture_radius = self
            .drivers
            .iter()
            .map(|driver| driver.radius_m)
            .reduce(f64::max)
            .ok_or(ProductDesignError::NoDrivers)?;
        let equivalent_radius = self.cabinet.equivalent_radius_m();
        if self.drivers.iter().any(|driver| {
            driver.position_m[2].abs() > 1.0e-12
                || (driver.position_m[0] * driver.position_m[0]
                    + driver.position_m[1] * driver.position_m[1])
                    .sqrt()
                    + driver.radius_m
                    > equivalent_radius
        }) {
            return Err(ProductDesignError::ReferenceDriverOutsideEquivalentFixture);
        }
        let fixture = self
            .cabinet
            .to_coupling_fixture(driver_aperture_radius, discretization)
            .map_err(|error| ProductDesignError::ReferenceCoupling(error.to_string()))?;
        FemBemCoupling::from_fixture(&fixture, speed_of_sound, density)
            .map_err(|error| ProductDesignError::ReferenceCoupling(error.to_string()))
    }

    /// Run the product-owned distributed FEM–BEM reference on a frequency
    /// sweep and retain only validated product observables.
    ///
    /// The current axisymmetric fixture lowering is intentionally restricted
    /// to a centred single driver. That restriction is explicit: off-axis and
    /// multi-driver products must use the production BEM path until a genuine
    /// 3-D product reference is available. Every returned point carries the
    /// full-order solver residual and is rejected when the solver did not
    /// retain residual evidence or exceeded the requested tolerance.
    pub fn solve_fem_bem_reference_sweep(
        &self,
        discretization: CylinderFixtureDiscretization,
        speed_of_sound: f64,
        density: f64,
        frequencies_hz: &[f64],
        input_voltage_rms: Complex64,
        solver_config: &LinearSolveConfig,
    ) -> Result<ProductReferenceSweep, ProductDesignError> {
        if frequencies_hz.is_empty() {
            return Err(ProductDesignError::ReferenceSweepEmpty);
        }
        let reference = self.build_fem_bem_reference(discretization, speed_of_sound, density)?;
        let mut points = Vec::with_capacity(frequencies_hz.len());
        for &frequency_hz in frequencies_hz {
            if !frequency_hz.is_finite() || frequency_hz <= 0.0 {
                return Err(ProductDesignError::InvalidFrequency);
            }
            let drive_velocities = self.piston_velocities_at(frequency_hz, input_voltage_rms)?;
            let drive_velocity = *drive_velocities
                .first()
                .ok_or(ProductDesignError::NoDrivers)?;
            let drive = if drive_velocities.len() == 1 {
                CouplingDrive::DriverVelocity(drive_velocity)
            } else {
                CouplingDrive::DriverVelocityProfiles(
                    self.drivers
                        .iter()
                        .zip(drive_velocities.iter().copied())
                        .map(|(driver, value)| DriverVelocityProfile {
                            value,
                            center_x_m: driver.position_m[0],
                            center_y_m: driver.position_m[1],
                            radius_m: driver.radius_m,
                        })
                        .collect(),
                )
            };
            let solution = reference
                .solve_frequency(frequency_hz, drive, solver_config)
                .map_err(|error| ProductDesignError::ReferenceCoupling(error.to_string()))?;
            let relative_residual = solution
                .relative_residual
                .ok_or(ProductDesignError::ReferenceResidualMissing { frequency_hz })?;
            if relative_residual > solver_config.tolerance {
                return Err(ProductDesignError::ReferenceResidualMismatch {
                    frequency_hz,
                    residual: relative_residual,
                    tolerance: solver_config.tolerance,
                });
            }
            points.push(ProductReferencePoint {
                frequency_hz,
                drive_velocity_rms_m_per_s: drive_velocity,
                driver_velocities_rms_m_per_s: drive_velocities,
                aperture_volume_velocity_m3_per_s: solution
                    .aperture_volume_velocity(&reference.aperture_element_areas()),
                mean_pressure_rms_pa: solution.mean_pressure(),
                mean_abs_pressure_pa: solution.mean_abs_pressure(),
                solver_iterations: solution.solver_iterations,
                relative_residual,
                monolithic_evidence: None,
                interface_evidence: None,
                thermal_evidence: None,
                structural_evidence: None,
            });
        }
        let sweep = ProductReferenceSweep {
            speed_of_sound_m_per_s: speed_of_sound,
            density_kg_per_m3: density,
            interior_dofs: reference.num_dofs(),
            aperture_elements: reference.num_aperture_elements(),
            points,
            driver_count: self.drivers.len(),
            rom_promotion: None,
            provenance: ArtifactProvenance::native_golden(),
            mesh_convergence: Vec::new(),
        };
        sweep.validate()?;
        Ok(sweep)
    }

    /// Solve the full-order product reference sweep and retain residual-gated
    /// reduced-order promotion evidence for the same FEM–BEM fixture.
    ///
    /// The ROM is a fidelity estimator, not a replacement for the physical
    /// product reference observables.  Its normalized unit-velocity drive is
    /// therefore independent of the electrical drive used by `sweep`; the
    /// residual and promotion decisions are scale-invariant for this linear
    /// reference problem.
    #[allow(clippy::too_many_arguments)]
    pub fn solve_fem_bem_reference_sweep_with_rom(
        &self,
        discretization: CylinderFixtureDiscretization,
        speed_of_sound: f64,
        density: f64,
        frequencies_hz: &[f64],
        input_voltage_rms: Complex64,
        solver_config: &LinearSolveConfig,
        policy: RomPromotionPolicy,
    ) -> Result<ProductReferenceSweep, ProductDesignError> {
        let mut sweep = self.solve_fem_bem_reference_sweep(
            discretization,
            speed_of_sound,
            density,
            frequencies_hz,
            input_voltage_rms,
            solver_config,
        )?;
        if self.drivers.len() != 1 {
            // The full-order distributed profile solve above is retained;
            // the current ROM correction stores one spatial RHS pattern and
            // must not be promoted for a multi-pattern product.
            return Ok(sweep);
        }
        let reference = self.build_fem_bem_reference(discretization, speed_of_sound, density)?;
        let rom_config = AcousticRomConfig::for_sweep(
            speed_of_sound,
            *frequencies_hz
                .last()
                .ok_or(ProductDesignError::ReferenceSweepEmpty)?,
        );
        let rom = reference
            .build_rom(
                &rom_config,
                CouplingDrive::DriverVelocity(Complex64::new(1.0e-3, 0.0)),
                solver_config,
            )
            .map_err(|error| ProductDesignError::ReferenceCoupling(error.to_string()))?
            .with_static_correction(solver_config)
            .map_err(|error| ProductDesignError::ReferenceCoupling(error.to_string()))?;
        let report = rom
            .solve_sweep_with_promotion(frequencies_hz, policy, solver_config)
            .map_err(|error| ProductDesignError::ReferenceCoupling(error.to_string()))?;
        sweep.rom_promotion = Some(ProductRomPromotionEvidence::from_report(&report));
        sweep.validate()?;
        Ok(sweep)
    }

    /// Evaluate linear cabinet-panel loading from the retained distributed
    /// cavity-pressure evidence.  This is deliberately a post-processing
    /// step: panel displacement/stress is derived from the same frequency
    /// sweep and remains available for product constraints without replacing
    /// the exterior acoustic boundary with a static approximation.
    pub fn reference_cabinet_panel_response(
        &self,
        sweep: &ProductReferenceSweep,
        max_mode_order: usize,
    ) -> Result<Vec<CabinetPanelEvaluation>, ProductDesignError> {
        self.validate()?;
        sweep.validate()?;
        if max_mode_order == 0 {
            return Err(ProductDesignError::InvalidReferenceEvidence(
                "cabinet panel response requires a positive mode order".to_owned(),
            ));
        }
        let cabinet = self.cabinet.to_cabinet()?;
        let pressures = sweep
            .points
            .iter()
            .map(|point| (point.frequency_hz, point.mean_pressure_rms_pa))
            .collect::<Vec<_>>();
        cabinet_panel_response(&pressures, &cabinet, max_mode_order)
            .map_err(|error| ProductDesignError::ReferenceCoupling(error.to_string()))
    }

    /// Apply the crossover network to a common RMS input voltage.
    pub fn drive_voltages_at(
        &self,
        frequency_hz: f64,
        input_voltage_rms: Complex64,
    ) -> Result<Vec<Complex64>, ProductDesignError> {
        self.validate()?;
        if !input_voltage_rms.re.is_finite() || !input_voltage_rms.im.is_finite() {
            return Err(ProductDesignError::InvalidVoltage);
        }
        self.crossover.validate(self.drivers.len())?;
        self.drivers
            .iter()
            .map(|driver| {
                self.crossover.branches[driver.crossover_branch]
                    .transfer_at(frequency_hz)
                    .map(|transfer| input_voltage_rms * transfer)
            })
            .collect()
    }

    /// Evaluate the optional product structural response from a modeling-RMS
    /// cavity/panel pressure. The speaker crate owns the BEM convention bridge.
    pub fn structural_response_at(
        &self,
        frequency_hz: f64,
        pressure_rms_pa: Complex64,
    ) -> Result<Option<StructuralAcousticResponse>, ProductDesignError> {
        self.validate()?;
        self.structural_acoustic
            .as_ref()
            .map(|feedback| {
                feedback
                    .response_at(frequency_hz, pressure_rms_pa)
                    .map_err(|error| {
                        ProductDesignError::InvalidStructuralFeedback(error.to_string())
                    })
            })
            .transpose()
    }

    /// Compute the uncoupled small-signal piston velocities produced by a
    /// common RMS input. The speaker BEM replaces this local mechanical load
    /// with its radiation matrix when a typed-driver solve is available; this
    /// method is the explicit prescribed-velocity fallback for product arrays.
    pub fn piston_velocities_at(
        &self,
        frequency_hz: f64,
        input_voltage_rms: Complex64,
    ) -> Result<Vec<Complex64>, ProductDesignError> {
        let voltages = self.drive_voltages_at(frequency_hz, input_voltage_rms)?;
        self.drivers
            .iter()
            .zip(voltages)
            .enumerate()
            .map(|(index, (driver, voltage))| {
                let denominator = driver.coil_impedance_ohm()
                    * driver.mechanical_impedance_n_s_per_m()
                    + Complex64::new(driver.force_factor_tm.powi(2), 0.0);
                if denominator.norm() <= f64::MIN_POSITIVE {
                    return Err(ProductDesignError::InvalidDriver {
                        index,
                        reason: "uncoupled driver velocity denominator is singular",
                    });
                }
                Ok(Complex64::new(driver.force_factor_tm, 0.0) * voltage / denominator)
            })
            .collect()
    }

    /// Flatten the cabinet and driver placement variables for an optimizer.
    pub fn design_variables(&self) -> Vec<super::DesignVariable> {
        let mut variables = self.cabinet.design_variables();
        for (index, driver) in self.drivers.iter().enumerate() {
            variables.push(super::DesignVariable {
                name: format!("driver_{index}_x_m"),
                value: driver.position_m[0],
                lower: -self.cabinet.baffle_width_m * 0.5 + driver.radius_m,
                upper: self.cabinet.baffle_width_m * 0.5 - driver.radius_m,
            });
            variables.push(super::DesignVariable {
                name: format!("driver_{index}_y_m"),
                value: driver.position_m[1],
                lower: -self.cabinet.baffle_height_m * 0.5 + driver.radius_m,
                upper: self.cabinet.baffle_height_m * 0.5 - driver.radius_m,
            });
        }
        for (index, branch) in self.crossover.branches.iter().enumerate() {
            variables.push(super::DesignVariable {
                name: format!("crossover[{index}].corner_frequency_hz"),
                value: branch.corner_frequency_hz,
                lower: (0.25 * branch.corner_frequency_hz).max(1.0),
                upper: (4.0 * branch.corner_frequency_hz).max(1.0),
            });
            variables.push(super::DesignVariable {
                name: format!("crossover[{index}].gain"),
                value: branch.gain,
                lower: (0.25 * branch.gain).max(0.0),
                upper: (4.0 * branch.gain).max(0.1),
            });
            variables.push(super::DesignVariable {
                name: format!("crossover[{index}].delay_s"),
                value: branch.delay_s,
                lower: 0.0,
                upper: (branch.delay_s + 0.005).max(0.005),
            });
        }
        variables
    }

    /// Rebuild a product from its flattened continuous variables.
    pub fn from_design_variables(&self, values: &[f64]) -> Result<Self, ProductDesignError> {
        let cabinet_count = self.cabinet.design_variables().len();
        let crossover_count = 3 * self.crossover.branches.len();
        let expected = cabinet_count + 2 * self.drivers.len() + crossover_count;
        if values.len() != expected {
            return Err(ProductDesignError::DesignVariableCountMismatch {
                expected,
                got: values.len(),
            });
        }
        let cabinet = self
            .cabinet
            .from_design_variables(&values[..cabinet_count])?;
        let mut design = self.clone();
        design.cabinet = cabinet;
        let mut cursor = cabinet_count;
        for driver in &mut design.drivers {
            driver.position_m[0] = values[cursor];
            driver.position_m[1] = values[cursor + 1];
            cursor += 2;
        }
        for branch in &mut design.crossover.branches {
            branch.corner_frequency_hz = values[cursor];
            branch.gain = values[cursor + 1];
            branch.delay_s = values[cursor + 2];
            cursor += 3;
        }
        design.validate()?;
        Ok(design)
    }

    /// Project a flattened product-design vector to its declared continuous
    /// bounds. The optimizer normally supplies normalized values that are
    /// already bounded; this helper makes active-bound probes explicit.
    pub fn project_design_variables(&self, values: &[f64]) -> Result<Vec<f64>, ProductDesignError> {
        let variables = self.design_variables();
        if values.len() != variables.len() {
            return Err(ProductDesignError::DesignVariableCountMismatch {
                expected: variables.len(),
                got: values.len(),
            });
        }
        values
            .iter()
            .zip(&variables)
            .map(|(value, variable)| {
                if !value.is_finite() {
                    return Err(ProductDesignError::InvalidTaylorProbe(
                        "probe values must be finite",
                    ));
                }
                Ok(value.clamp(variable.lower, variable.upper))
            })
            .collect()
    }

    /// Measure a scalar Taylor remainder using projected, bound-aware probes.
    /// Product optimization uses an explicit bounded derivative-free policy;
    /// this validates smoothness without exposing an unverified gradient.
    pub fn bounded_taylor_error<F>(
        &self,
        values: &[f64],
        direction: &[f64],
        step: f64,
        evaluate: F,
    ) -> Result<f64, ProductDesignError>
    where
        F: Fn(&ProductDesign) -> Result<f64, ProductDesignError>,
    {
        if !step.is_finite() || step <= 0.0 {
            return Err(ProductDesignError::InvalidTaylorProbe(
                "probe step must be finite and positive",
            ));
        }
        if direction.len() != values.len()
            || direction.iter().any(|value| !value.is_finite())
            || direction.iter().all(|value| value.abs() <= f64::EPSILON)
        {
            return Err(ProductDesignError::InvalidTaylorProbe(
                "probe direction must be finite and non-zero",
            ));
        }
        let base_values = self.project_design_variables(values)?;
        let base = self.from_design_variables(&base_values)?;
        let f0 = evaluate(&base)?;
        if !f0.is_finite() {
            return Err(ProductDesignError::InvalidTaylorProbe(
                "base objective must be finite",
            ));
        }
        let project_step = |multiple: f64| -> Result<(Vec<f64>, f64), ProductDesignError> {
            let raw = base_values
                .iter()
                .zip(direction)
                .map(|(value, delta)| value + multiple * step * delta)
                .collect::<Vec<_>>();
            let projected = self.project_design_variables(&raw)?;
            let denominator = direction.iter().map(|value| value * value).sum::<f64>();
            let effective_step = projected
                .iter()
                .zip(&base_values)
                .zip(direction)
                .map(|((value, base), direction)| (value - base) * direction)
                .sum::<f64>()
                / denominator;
            Ok((projected, effective_step))
        };
        let (plus_values, plus_step) = project_step(1.0)?;
        let (minus_values, minus_step) = project_step(-1.0)?;
        let plus = (plus_step.abs() > f64::EPSILON)
            .then(|| evaluate(&self.from_design_variables(&plus_values)?))
            .transpose()?;
        let minus = (minus_step.abs() > f64::EPSILON)
            .then(|| evaluate(&self.from_design_variables(&minus_values)?))
            .transpose()?;
        let (slope, mut remainder) = match (plus, minus) {
            (Some(plus), Some(minus)) => {
                let slope = (plus - minus) / (plus_step - minus_step);
                (slope, (plus - f0 - slope * plus_step).abs())
            }
            (Some(plus), None) => ((plus - f0) / plus_step, 0.0),
            (None, Some(minus)) => ((f0 - minus) / (-minus_step), 0.0),
            (None, None) => {
                return Err(ProductDesignError::InvalidTaylorProbe(
                    "projection removed both Taylor probe directions",
                ));
            }
        };
        let (plus2_values, plus2_step) = project_step(2.0)?;
        if plus2_step.abs() > plus_step.abs() + f64::EPSILON {
            let value = evaluate(&self.from_design_variables(&plus2_values)?)?;
            remainder = remainder.max((value - f0 - slope * plus2_step).abs());
        }
        let (minus2_values, minus2_step) = project_step(-2.0)?;
        if minus2_step.abs() > minus_step.abs() + f64::EPSILON {
            let value = evaluate(&self.from_design_variables(&minus2_values)?)?;
            remainder = remainder.max((value - f0 - slope * minus2_step).abs());
        }
        Ok(remainder / f0.abs().max(1.0))
    }

    /// Lower a centred single-driver product to the resolved axisymmetric
    /// acoustic model.  Off-axis and multi-driver products must use the full
    /// BEM/multi-port path instead of silently losing placement information.
    pub fn to_resolved_config(&self) -> Result<super::ResolvedAcousticConfig, ProductDesignError> {
        self.validate()?;
        if self.drivers.len() != 1 {
            return Err(ProductDesignError::ResolvedModelRequiresSingleDriver);
        }
        let driver = &self.drivers[0];
        if driver.position_m[0].abs() > 1.0e-12 || driver.position_m[1].abs() > 1.0e-12 {
            return Err(ProductDesignError::ResolvedModelRequiresCentredDriver);
        }
        if !driver.is_axis_aligned() || driver.normal != [0.0, 0.0, 1.0] {
            return Err(ProductDesignError::ResolvedModelRequiresFrontDriver);
        }
        Ok(self.cabinet.to_resolved_config(driver.radius_m)?)
    }
}

impl ProductDesign {
    /// Build structural promotion evidence from this product's configured
    /// reduced structural-feedback model.
    pub fn structural_reference_evidence(
        &self,
        frequency_hz: f64,
        pressure_rms_pa: Complex64,
        iterations: usize,
        relative_velocity_residual: f64,
        mesh_convergence_error: f64,
    ) -> Result<ProductStructuralReferenceEvidence, ProductDesignError> {
        self.validate()?;
        let feedback = self.structural_acoustic.as_ref().ok_or_else(|| {
            ProductDesignError::InvalidStructuralFeedback(
                "product has no structural-acoustic feedback model".into(),
            )
        })?;
        ProductStructuralReferenceEvidence::from_feedback(
            feedback,
            frequency_hz,
            pressure_rms_pa,
            iterations,
            relative_velocity_residual,
            mesh_convergence_error,
        )
    }

    /// Convert monolithic electromechanical losses into validated per-driver
    /// steady thermal evidence.
    pub fn thermal_reference_evidence(
        &self,
        solution: &crate::fem_bem_coupling::MonolithicSealedProductSolution,
        config: DriverThermalNetworkConfig,
        eddy_loss_w: &[f64],
    ) -> Result<ProductThermalReferenceEvidence, ProductDesignError> {
        self.validate()?;
        if solution.driver_current_a.len() != self.drivers.len()
            || eddy_loss_w.len() != self.drivers.len()
        {
            return Err(ProductDesignError::InvalidReferenceEvidence(
                "thermal loss evidence driver count differs from product".into(),
            ));
        }

        let mut source_power_w = Vec::with_capacity(self.drivers.len());
        let mut states = Vec::with_capacity(self.drivers.len());
        for ((driver, current), &eddy_loss) in self
            .drivers
            .iter()
            .zip(&solution.driver_current_a)
            .zip(eddy_loss_w)
        {
            let copper_loss = driver.coil_resistance_ohm * current.norm_sqr();
            let power = config
                .power_from_losses(copper_loss, eddy_loss)
                .map_err(|error| {
                    ProductDesignError::InvalidReferenceEvidence(format!(
                        "invalid product thermal source: {error}"
                    ))
                })?;
            let state = config
                .specification
                .solve_steady_state(power)
                .map_err(|error| {
                    ProductDesignError::InvalidReferenceEvidence(format!(
                        "product thermal solve failed: {error}"
                    ))
                })?;
            source_power_w.push(power);
            states.push(state);
        }

        let evidence = ProductThermalReferenceEvidence {
            schema_version: PRODUCT_THERMAL_EVIDENCE_SCHEMA_VERSION,
            driver_ids: self
                .drivers
                .iter()
                .map(|driver| driver.id.clone())
                .collect(),
            source_power_w,
            states,
            hot_coil_resistance_ohm: None,
            feedback_iterations: None,
            feedback_relative_resistance_change: None,
        };
        evidence.validate_for_driver_ids(
            &self
                .drivers
                .iter()
                .map(|driver| driver.id.clone())
                .collect::<Vec<_>>(),
            ProductThermalPromotionGate::default(),
        )?;
        Ok(evidence)
    }

    /// Build a portable product reference point with thermal evidence from
    /// the same monolithic electromechanical state.
    /// Convert a thermally converged electromechanical state into evidence,
    /// retaining the hot copper resistance used by the acoustic solve.
    pub fn thermal_reference_evidence_with_feedback(
        &self,
        solution: &crate::fem_bem_coupling::MonolithicSealedProductSolution,
        config: DriverThermalNetworkConfig,
        eddy_loss_w: &[f64],
        hot_coil_resistance_ohm: &[f64],
        feedback_iterations: usize,
        feedback_relative_resistance_change: f64,
    ) -> Result<ProductThermalReferenceEvidence, ProductDesignError> {
        self.validate()?;
        if solution.driver_current_a.len() != self.drivers.len()
            || eddy_loss_w.len() != self.drivers.len()
            || hot_coil_resistance_ohm.len() != self.drivers.len()
            || feedback_iterations == 0
            || !feedback_relative_resistance_change.is_finite()
            || feedback_relative_resistance_change < 0.0
        {
            return Err(ProductDesignError::InvalidReferenceEvidence(
                "invalid thermal feedback evidence shape or convergence metadata".into(),
            ));
        }
        let mut source_power_w = Vec::with_capacity(self.drivers.len());
        let mut states = Vec::with_capacity(self.drivers.len());
        for ((&current, &eddy_loss), &coil_resistance) in solution
            .driver_current_a
            .iter()
            .zip(eddy_loss_w)
            .zip(hot_coil_resistance_ohm)
        {
            if !coil_resistance.is_finite() || coil_resistance <= 0.0 {
                return Err(ProductDesignError::InvalidReferenceEvidence(
                    "thermal feedback hot copper resistance must be finite positive".into(),
                ));
            }
            let power = config
                .power_from_losses(coil_resistance * current.norm_sqr(), eddy_loss)
                .map_err(|error| {
                    ProductDesignError::InvalidReferenceEvidence(format!(
                        "invalid product thermal feedback source: {error}"
                    ))
                })?;
            let state = config
                .specification
                .solve_steady_state(power)
                .map_err(|error| {
                    ProductDesignError::InvalidReferenceEvidence(format!(
                        "product thermal feedback solve failed: {error}"
                    ))
                })?;
            source_power_w.push(power);
            states.push(state);
        }
        let evidence = ProductThermalReferenceEvidence {
            schema_version: PRODUCT_THERMAL_EVIDENCE_SCHEMA_VERSION,
            driver_ids: self
                .drivers
                .iter()
                .map(|driver| driver.id.clone())
                .collect(),
            source_power_w,
            states,
            hot_coil_resistance_ohm: Some(hot_coil_resistance_ohm.to_vec()),
            feedback_iterations: Some(feedback_iterations),
            feedback_relative_resistance_change: Some(feedback_relative_resistance_change),
        };
        evidence.validate_for_driver_ids(
            &self
                .drivers
                .iter()
                .map(|driver| driver.id.clone())
                .collect::<Vec<_>>(),
            ProductThermalPromotionGate::default(),
        )?;
        Ok(evidence)
    }

    pub fn reference_point_from_monolithic_solution_with_thermal(
        &self,
        solution: &crate::fem_bem_coupling::MonolithicSealedProductSolution,
        mesh: &crate::product_acoustic_mesh::ProductInteriorMesh,
        mesh_convergence_error: f64,
        config: DriverThermalNetworkConfig,
        eddy_loss_w: &[f64],
    ) -> Result<ProductReferencePoint, ProductDesignError> {
        let mut point = ProductReferencePoint::from_monolithic_solution_with_structural_feedback(
            solution,
            mesh,
            mesh_convergence_error,
            self.structural_acoustic.as_ref(),
        )?;
        point.thermal_evidence =
            Some(self.thermal_reference_evidence(solution, config, eddy_loss_w)?);
        point.validate()?;
        Ok(point)
    }
}

/// Errors in the product-level contract.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum ProductDesignError {
    #[error(transparent)]
    Cabinet(#[from] CabinetDesignError),
    #[error("product must contain at least one driver")]
    NoDrivers,
    #[error("crossover branch count mismatch: expected {expected}, got {got}")]
    CrossoverCountMismatch { expected: usize, got: usize },
    #[error("invalid crossover: {0}")]
    InvalidCrossover(&'static str),
    #[error("frequency must be finite and positive")]
    InvalidFrequency,
    #[error("input voltage must be finite")]
    InvalidVoltage,
    #[error("driver {index} is invalid: {reason}")]
    InvalidDriver { index: usize, reason: &'static str },
    #[error("driver {index} lies outside the baffle")]
    DriverOutsideBaffle { index: usize },
    #[error("product contains duplicate driver id {id:?}")]
    DuplicateDriverId { id: String },
    #[error(
        "crossover branch {branch} is assigned to both drivers {first_driver} and {second_driver}; product branches must map one-to-one to drivers"
    )]
    CrossoverBranchReuse {
        branch: usize,
        first_driver: usize,
        second_driver: usize,
    },
    #[error("drivers {left} and {right} overlap")]
    DriversOverlap { left: usize, right: usize },
    #[error("port overlaps driver {index}")]
    PortOverlapsDriver { index: usize },
    #[error("invalid port boundary impedance: {0}")]
    InvalidPortBoundary(String),
    #[error("invalid product waveguide: {0}")]
    InvalidWaveguide(String),
    #[error("the FEM-BEM reference model requires exactly one driver")]
    ReferenceModelRequiresSingleDriver,
    #[error("the FEM-BEM reference model requires a centred driver")]
    ReferenceModelRequiresCentredDriver,
    #[error("the FEM-BEM reference model requires a centred port")]
    ReferenceModelRequiresCentredPort,
    #[error("a reference driver lies outside the equivalent cylindrical fixture")]
    ReferenceDriverOutsideEquivalentFixture,
    #[error("product FEM-BEM reference construction failed: {0}")]
    ReferenceCoupling(String),
    #[error("product FEM-BEM reference sweep must contain at least one frequency")]
    ReferenceSweepEmpty,
    #[error("product FEM-BEM reference has no retained residual at {frequency_hz} Hz")]
    ReferenceResidualMissing { frequency_hz: f64 },
    #[error(
        "product FEM-BEM reference residual {residual:.3e} exceeds tolerance {tolerance:.3e} at {frequency_hz} Hz"
    )]
    ReferenceResidualMismatch {
        frequency_hz: f64,
        residual: f64,
        tolerance: f64,
    },
    #[error("invalid product FEM-BEM reference evidence: {0}")]
    InvalidReferenceEvidence(String),
    #[error("invalid structural acoustic feedback: {0}")]
    InvalidStructuralFeedback(String),
    #[error("invalid product design artifact: {0}")]
    InvalidProductArtifact(String),
    #[error("invalid bounded Taylor probe: {0}")]
    InvalidTaylorProbe(&'static str),
    #[error("product design variable count mismatch: expected {expected}, got {got}")]
    DesignVariableCountMismatch { expected: usize, got: usize },
    #[error("resolved axisymmetric lowering requires exactly one driver")]
    ResolvedModelRequiresSingleDriver,
    #[error("resolved axisymmetric lowering requires a centred driver")]
    ResolvedModelRequiresCentredDriver,
    #[error("resolved axisymmetric lowering requires a front-facing axis-aligned driver")]
    ResolvedModelRequiresFrontDriver,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::port_flow::{
        PORT_FLOW_PROMOTION_SCHEMA_VERSION, PortFlowConvergenceLevel, PortFlowPhasor,
        PortFlowRuntimePlan,
    };
    use crate::structural_acoustic::StructuralAcousticMode;
    use crate::thermal_network::{DriverThermalNetworkConfig, ThermalNetworkPower};
    use crate::{CabinetMaterial, ProductWaveguideDesign};

    fn driver(id: &str, x: f64, y: f64, branch: usize) -> ProductDriver {
        ProductDriver {
            id: id.to_owned(),
            radius_m: 0.025,
            position_m: [x, y, 0.0],
            normal: [0.0, 0.0, 1.0],
            force_factor_tm: 5.0,
            coil_resistance_ohm: 5.0,
            coil_reactance_ohm: 1.0,
            mechanical_resistance_n_s_per_m: 0.5,
            mechanical_reactance_n_s_per_m: 2.0,
            crossover_branch: branch,
        }
    }

    fn product() -> ProductDesign {
        ProductDesign {
            cabinet: CabinetDesign::default(),
            drivers: vec![
                driver("woofer", -0.07, -0.12, 0),
                driver("tweeter", 0.07, 0.12, 1),
            ],
            crossover: CrossoverNetwork {
                branches: vec![
                    CrossoverBranch {
                        kind: CrossoverKind::LowPass,
                        corner_frequency_hz: 2_000.0,
                        ..CrossoverBranch::default()
                    },
                    CrossoverBranch {
                        kind: CrossoverKind::HighPass,
                        corner_frequency_hz: 2_000.0,
                        ..CrossoverBranch::default()
                    },
                ],
            },
            waveguide: None,
            structural_acoustic: None,
        }
    }

    #[test]
    fn product_contract_validates_placement_and_crossover() {
        let design = product();
        design.validate().unwrap();
        let voltages = design
            .drive_voltages_at(2_000.0, Complex64::new(2.0, 0.0))
            .unwrap();
        assert_eq!(voltages.len(), 2);
        assert!(voltages[0].norm() > 0.0);
        assert!(voltages[1].norm() > 0.0);
        assert_ne!(voltages[0].im, voltages[1].im);
    }

    #[test]
    fn product_design_builds_body_fitted_monolithic_reference() {
        let mut design = ProductDesign {
            cabinet: CabinetDesign {
                port: Some({
                    let mut port = CabinetDesign::default().port.expect("default port");
                    port.position_m = [0.06, 0.0];
                    port
                }),
                ..CabinetDesign::default()
            },
            drivers: vec![driver("woofer", -0.06, 0.0, 0)],
            crossover: CrossoverNetwork {
                branches: vec![CrossoverBranch::default()],
            },
            waveguide: None,
            structural_acoustic: None,
        };
        design.drivers[0].radius_m = 0.045;
        design.drivers[0].position_m = [-0.06, 0.0, 0.0];

        let coupling = design
            .build_body_fitted_monolithic_reference(
                ProductInteriorMeshConfig {
                    max_element_size_m: 0.08,
                    min_cells_per_axis: 2,
                    max_cells_per_axis: 32,
                },
                343.0,
                1.204,
            )
            .expect("body-fitted product coupling should build");

        let mesh = coupling.interior_mesh();
        assert!(mesh.geometry_fingerprint.contains("body-fitted-driver"));
        assert!(
            mesh.patch(ProductInteriorPatchKind::DriverRear { driver_index: 0 })
                .is_some()
        );
        assert_eq!(mesh.diagnostics.node_count, mesh.mesh.num_nodes());
        assert!(matches!(
            design.build_body_fitted_monolithic_reference(
                ProductInteriorMeshConfig::default(),
                f64::NAN,
                1.204,
            ),
            Err(ProductDesignError::InvalidReferenceEvidence(_))
        ));
    }

    #[test]
    fn product_design_solves_body_fitted_monolithic_reference_sweep_with_thermal() {
        let mut design = product();
        design.drivers.truncate(1);
        design.crossover.branches.truncate(1);

        let sweep = design
            .solve_body_fitted_monolithic_reference_sweep_with_thermal(
                ProductInteriorMeshConfig {
                    max_element_size_m: 0.08,
                    min_cells_per_axis: 2,
                    max_cells_per_axis: 32,
                },
                343.0,
                1.204,
                &[120.0],
                Complex64::new(2.0, 0.0),
                DriverThermalNetworkConfig::default(),
                &[vec![0.25]],
            )
            .expect("thermal body-fitted reference sweep");

        let point = &sweep.points[0];
        let thermal = point.thermal_evidence.as_ref().expect("thermal evidence");
        assert_eq!(thermal.driver_ids, vec!["woofer"]);
        assert_eq!(thermal.states.len(), 1);
        assert!(thermal.states[0].temperatures_k[0].is_finite());

        let gate = MonolithicEvidenceGate {
            require_thermal_evidence: true,
            ..MonolithicEvidenceGate::default()
        };
        sweep
            .validate_monolithic_product_evidence(gate)
            .expect("thermal evidence gate");
    }

    #[test]
    fn product_design_closes_thermal_copper_feedback_and_retains_hot_resistance() {
        let mut design = product();
        design.drivers.truncate(1);
        design.crossover.branches.truncate(1);
        let sweep = design
            .solve_body_fitted_monolithic_reference_sweep_with_thermal_feedback(
                ProductInteriorMeshConfig {
                    max_element_size_m: 0.08,
                    min_cells_per_axis: 2,
                    max_cells_per_axis: 32,
                },
                343.0,
                1.204,
                &[120.0],
                Complex64::new(2.0, 0.0),
                DriverThermalNetworkConfig::default(),
                &[vec![0.25]],
                64,
                1.0e-8,
            )
            .expect("thermal copper feedback should converge");
        let thermal = sweep.points[0]
            .thermal_evidence
            .as_ref()
            .expect("feedback thermal evidence");
        let hot_resistance = thermal
            .hot_coil_resistance_ohm
            .as_ref()
            .expect("hot resistance metadata")[0];
        assert!(hot_resistance > design.drivers[0].coil_resistance_ohm);
        assert!(thermal.feedback_iterations.unwrap_or(0) > 0);
        assert!(
            thermal
                .feedback_relative_resistance_change
                .expect("feedback residual")
                <= 1.0e-8
        );
        sweep
            .validate_monolithic_product_evidence(MonolithicEvidenceGate {
                require_thermal_evidence: true,
                ..MonolithicEvidenceGate::default()
            })
            .expect("feedback evidence should pass the thermal gate");

        let encoded = serde_json::to_string(&sweep).expect("serialize thermal feedback sweep");
        let decoded: ProductReferenceSweep =
            serde_json::from_str(&encoded).expect("deserialize thermal feedback sweep");
        let decoded_thermal = decoded.points[0]
            .thermal_evidence
            .as_ref()
            .expect("decoded feedback thermal evidence");
        assert_eq!(
            decoded_thermal.hot_coil_resistance_ohm,
            thermal.hot_coil_resistance_ohm
        );
        assert_eq!(
            decoded_thermal.feedback_iterations,
            thermal.feedback_iterations
        );
        assert_eq!(decoded_thermal.driver_ids, thermal.driver_ids);
        decoded
            .validate()
            .expect("decoded feedback sweep validates");
    }

    #[test]
    fn product_design_thermal_feedback_rejects_invalid_budget_and_nonconvergence() {
        let mut design = product();
        design.drivers.truncate(1);
        design.crossover.branches.truncate(1);
        let args = ProductInteriorMeshConfig {
            max_element_size_m: 0.08,
            min_cells_per_axis: 2,
            max_cells_per_axis: 32,
        };
        let invalid = design
            .solve_body_fitted_monolithic_reference_sweep_with_thermal_feedback(
                args,
                343.0,
                1.204,
                &[120.0],
                Complex64::new(2.0, 0.0),
                DriverThermalNetworkConfig::default(),
                &[vec![0.25]],
                0,
                1.0e-8,
            )
            .expect_err("zero thermal feedback budget must fail closed");
        assert!(invalid.to_string().contains("iteration budget"));

        let nonconverged = design
            .solve_body_fitted_monolithic_reference_sweep_with_thermal_feedback(
                args,
                343.0,
                1.204,
                &[120.0],
                Complex64::new(2.0, 0.0),
                DriverThermalNetworkConfig::default(),
                &[vec![0.25]],
                1,
                1.0e-16,
            )
            .expect_err("one iteration must not claim a fixed point");
        assert!(nonconverged.to_string().contains("did not converge"));
    }

    #[test]
    fn monolithic_product_solver_uses_supplied_hot_coil_resistance() {
        let mut design = product();
        design.drivers.truncate(1);
        design.crossover.branches.truncate(1);
        let coupling = design
            .build_body_fitted_monolithic_reference(
                ProductInteriorMeshConfig {
                    max_element_size_m: 0.08,
                    min_cells_per_axis: 2,
                    max_cells_per_axis: 32,
                },
                343.0,
                1.204,
            )
            .expect("body-fitted coupling");
        let cold = coupling
            .solve_frequency(&design, 120.0, Complex64::new(2.0, 0.0))
            .expect("cold resistance solve");
        let hot = coupling
            .solve_frequency_with_coil_resistances(
                &design,
                120.0,
                Complex64::new(2.0, 0.0),
                Some(&[10.0]),
            )
            .expect("hot resistance solve");
        assert!((cold.driver_current_a[0] - hot.driver_current_a[0]).norm() > 1.0e-8);
        assert!(
            (hot.coil_dissipation_w - 10.0 * hot.driver_current_a[0].norm_sqr()).abs() < 1.0e-10
        );
    }

    #[test]
    fn body_fitted_thermal_sweep_rejects_malformed_loss_grid_and_config() {
        let mut design = product();
        design.drivers.truncate(1);
        design.crossover.branches.truncate(1);
        let mesh_config = ProductInteriorMeshConfig {
            max_element_size_m: 0.025,
            min_cells_per_axis: 2,
            max_cells_per_axis: 256,
        };

        assert!(matches!(
            design.solve_body_fitted_monolithic_reference_sweep_with_thermal(
                mesh_config,
                343.0,
                1.204,
                &[120.0, 240.0],
                Complex64::new(2.0, 0.0),
                DriverThermalNetworkConfig::default(),
                &[vec![0.25]],
            ),
            Err(ProductDesignError::InvalidReferenceEvidence(message))
                if message.contains("one row per frequency")
        ));

        assert!(matches!(
            design.solve_body_fitted_monolithic_reference_sweep_with_thermal(
                mesh_config,
                343.0,
                1.204,
                &[120.0],
                Complex64::new(2.0, 0.0),
                DriverThermalNetworkConfig::default(),
                &[vec![]],
            ),
            Err(ProductDesignError::InvalidReferenceEvidence(message))
                if message.contains("driver count")
        ));

        let mut invalid_config = DriverThermalNetworkConfig::default();
        invalid_config.eddy_loss_to_magnet_fraction = 1.1;
        assert!(matches!(
            design.solve_body_fitted_monolithic_reference_sweep_with_thermal(
                mesh_config,
                343.0,
                1.204,
                &[120.0],
                Complex64::new(2.0, 0.0),
                invalid_config,
                &[vec![0.25]],
            ),
            Err(ProductDesignError::InvalidReferenceEvidence(message))
                if message.contains("thermal configuration")
        ));
    }

    #[test]
    fn body_fitted_thermal_mesh_ladder_retains_thermal_points() {
        let mut design = product();
        design.drivers.truncate(1);
        design.crossover.branches.truncate(1);
        let level = |label: &str, size_m: f64| ProductBodyFittedMeshLevel {
            label: label.into(),
            mesh_config: ProductInteriorMeshConfig {
                max_element_size_m: size_m,
                min_cells_per_axis: 2,
                max_cells_per_axis: 32,
            },
        };

        let sweep = design
            .solve_body_fitted_monolithic_mesh_convergence_sweep_with_thermal(
                &[
                    level("coarse", 0.01),
                    level("medium", 0.008),
                    level("fine", 0.006),
                ],
                343.0,
                1.204,
                &[120.0],
                Complex64::new(2.0, 0.0),
                DriverThermalNetworkConfig::default(),
                &[vec![0.25]],
            )
            .expect("thermal body-fitted mesh ladder");

        assert_eq!(sweep.mesh_convergence.len(), 1);
        assert_eq!(sweep.mesh_convergence[0].levels.len(), 3);
        assert!(
            sweep.mesh_convergence[0]
                .levels
                .iter()
                .all(|level| level.point.thermal_evidence.is_some())
        );
        sweep.validate().expect("thermal mesh ladder evidence");
    }

    #[test]
    fn product_design_solves_body_fitted_monolithic_reference_sweep() {
        let mut design = product();
        design.drivers.truncate(1);
        design.crossover.branches.truncate(1);

        let sweep = design
            .solve_body_fitted_monolithic_reference_sweep(
                ProductInteriorMeshConfig {
                    max_element_size_m: 0.08,
                    min_cells_per_axis: 2,
                    max_cells_per_axis: 32,
                },
                343.0,
                1.204,
                &[120.0],
                Complex64::new(2.0, 0.0),
            )
            .expect("body-fitted product reference sweep");

        assert_eq!(sweep.points.len(), 1);
        assert_eq!(sweep.driver_count, 1);
        assert!(sweep.interior_dofs > 0);
        assert!(sweep.aperture_elements > 0);
        assert!(sweep.rom_promotion.is_none());
        assert!(sweep.mesh_convergence.is_empty());
        assert_eq!(sweep.points[0].frequency_hz, 120.0);
        assert_eq!(
            sweep.points[0]
                .monolithic_evidence
                .as_ref()
                .expect("monolithic evidence")
                .mesh_node_count,
            sweep.interior_dofs
        );
        assert!(
            sweep.points[0]
                .interface_evidence
                .as_ref()
                .is_some_and(|interfaces| interfaces
                    .iter()
                    .any(|interface| interface.interface_id == "driver-aperture-0"))
        );
    }

    #[test]
    fn body_fitted_monolithic_reference_sweep_rejects_malformed_inputs() {
        let mut design = product();
        design.drivers.truncate(1);
        design.crossover.branches.truncate(1);
        let mesh_config = ProductInteriorMeshConfig {
            max_element_size_m: 0.08,
            min_cells_per_axis: 2,
            max_cells_per_axis: 32,
        };

        assert!(matches!(
            design.solve_body_fitted_monolithic_reference_sweep(
                mesh_config,
                343.0,
                1.204,
                &[],
                Complex64::new(2.0, 0.0),
            ),
            Err(ProductDesignError::ReferenceSweepEmpty)
        ));
        assert!(matches!(
            design.solve_body_fitted_monolithic_reference_sweep(
                mesh_config,
                343.0,
                1.204,
                &[120.0],
                Complex64::new(f64::NAN, 0.0),
            ),
            Err(ProductDesignError::InvalidVoltage)
        ));
        assert!(matches!(
            design.solve_body_fitted_monolithic_reference_sweep(
                mesh_config,
                343.0,
                1.204,
                &[f64::NAN],
                Complex64::new(2.0, 0.0),
            ),
            Err(ProductDesignError::InvalidFrequency)
        ));
    }

    #[test]
    fn product_design_executes_body_fitted_mesh_convergence_ladder() {
        let mut design = product();
        design.drivers.truncate(1);
        design.crossover.branches.truncate(1);
        let mesh_levels = vec![
            ProductBodyFittedMeshLevel {
                label: "coarse".into(),
                mesh_config: ProductInteriorMeshConfig {
                    max_element_size_m: 0.01,
                    min_cells_per_axis: 2,
                    max_cells_per_axis: 32,
                },
            },
            ProductBodyFittedMeshLevel {
                label: "medium".into(),
                mesh_config: ProductInteriorMeshConfig {
                    max_element_size_m: 0.008,
                    min_cells_per_axis: 2,
                    max_cells_per_axis: 32,
                },
            },
            ProductBodyFittedMeshLevel {
                label: "fine".into(),
                mesh_config: ProductInteriorMeshConfig {
                    max_element_size_m: 0.006,
                    min_cells_per_axis: 2,
                    max_cells_per_axis: 32,
                },
            },
        ];

        let sweep = design
            .solve_body_fitted_monolithic_mesh_convergence_sweep(
                &mesh_levels,
                343.0,
                1.204,
                &[120.0],
                Complex64::new(2.0, 0.0),
            )
            .expect("body-fitted product mesh-convergence sweep");

        assert_eq!(sweep.points.len(), 1);
        assert_eq!(sweep.mesh_convergence.len(), 1);
        let ladder = &sweep.mesh_convergence[0];
        assert_eq!(
            ladder
                .levels
                .iter()
                .map(|level| level.label.as_str())
                .collect::<Vec<_>>(),
            vec!["coarse", "medium", "fine"]
        );
        assert_eq!(ladder.adjacent_observable_errors.len(), 2);
        assert!(ladder.maximum_adjacent_observable_error.is_finite());
        assert_eq!(
            ladder.finest_point().unwrap().monolithic_evidence.as_ref(),
            sweep.points[0].monolithic_evidence.as_ref()
        );
        sweep.validate().expect("retained body-fitted ladder");
    }

    #[test]
    fn body_fitted_mesh_convergence_rejects_invalid_level_contract() {
        let mut design = product();
        design.drivers.truncate(1);
        design.crossover.branches.truncate(1);
        let level = |label: &str, size_m: f64| ProductBodyFittedMeshLevel {
            label: label.into(),
            mesh_config: ProductInteriorMeshConfig {
                max_element_size_m: size_m,
                min_cells_per_axis: 2,
                max_cells_per_axis: 32,
            },
        };

        assert!(matches!(
            design.solve_body_fitted_monolithic_mesh_convergence_sweep(
                &[level("coarse", 0.12), level("medium", 0.08)],
                343.0,
                1.204,
                &[120.0],
                Complex64::new(2.0, 0.0),
            ),
            Err(ProductDesignError::InvalidReferenceEvidence(_))
        ));
        assert!(matches!(
            design.solve_body_fitted_monolithic_mesh_convergence_sweep(
                &[
                    level("same", 0.12),
                    level("same", 0.08),
                    level("fine", 0.06)
                ],
                343.0,
                1.204,
                &[120.0],
                Complex64::new(2.0, 0.0),
            ),
            Err(ProductDesignError::InvalidReferenceEvidence(_))
        ));
        assert!(matches!(
            design.solve_body_fitted_monolithic_mesh_convergence_sweep(
                &[
                    level("coarse", 0.12),
                    level("medium", 0.08),
                    level("fine", 0.06)
                ],
                343.0,
                1.204,
                &[120.0, 120.0],
                Complex64::new(2.0, 0.0),
            ),
            Err(ProductDesignError::InvalidReferenceEvidence(_))
        ));
    }

    #[test]
    fn port_boundary_impedance_is_volume_flow_passive_and_resonant() {
        let design = product();
        let medium = AcousticMedium::standard_air();
        let low = design
            .port_boundary_impedance_at(20.0, medium)
            .unwrap()
            .unwrap();
        let high = design
            .port_boundary_impedance_at(200.0, medium)
            .unwrap()
            .unwrap();
        assert!(low.re >= 0.0);
        assert!(high.re >= 0.0);
        assert!(low.im < 0.0, "below tuning the cavity is compliant: {low}");
        assert!(high.im > 0.0, "above tuning the port is mass-like: {high}");

        let tuning = design
            .cabinet
            .to_lumped()
            .unwrap()
            .tuning_frequency_hz(medium)
            .unwrap()
            .unwrap();
        let at_tuning = design
            .port_boundary_impedance_at(tuning, medium)
            .unwrap()
            .unwrap();
        assert!(at_tuning.im.abs() < 1.0e-8, "tuning impedance {at_tuning}");
    }

    #[test]
    fn product_accepts_axis_aligned_and_bounded_tilted_cabinet_faces() {
        let mut tilted = product();
        tilted.drivers[0].normal = [0.1, 0.0, (1.0 - 0.1_f64.powi(2)).sqrt()];
        assert!(tilted.validate().is_ok());
        let frame = tilted.drivers[0]
            .aperture_frame(
                tilted.cabinet.baffle_width_m,
                tilted.cabinet.baffle_height_m,
                tilted.cabinet.box_width_m,
                tilted.cabinet.box_height_m,
                tilted.cabinet.box_depth_m,
            )
            .expect("bounded tilted aperture frame");
        assert_eq!(frame.mounting_face, DriverMountingFace::Front);
        assert!(frame.inward_offset_m > 0.0);
        assert!(frame.projected_radius_m > tilted.drivers[0].radius_m);
        let projected = frame.project_to_wall(frame.aperture_point(0.37));
        assert!((projected[2] - tilted.drivers[0].position_m[2]).abs() < 1.0e-12);

        let mut back = product();
        back.drivers[0].normal = [0.0, 0.0, -1.0];
        back.drivers[0].position_m[2] = -back.cabinet.box_depth_m;
        assert!(back.validate().is_ok());

        let mut right = product();
        right.drivers[0].normal = [1.0, 0.0, 0.0];
        right.drivers[0].position_m = [
            right.cabinet.box_width_m * 0.5,
            0.0,
            -right.cabinet.box_depth_m * 0.5,
        ];
        assert!(right.validate().is_ok());
        assert!(matches!(
            right.build_fem_bem_reference(CylinderFixtureDiscretization::default(), 343.0, 1.204),
            Err(ProductDesignError::ReferenceDriverOutsideEquivalentFixture)
        ));

        let mut inward = right.clone();
        inward.drivers[0].normal = [-1.0, 0.0, 0.0];
        assert!(matches!(
            inward.validate(),
            Err(ProductDesignError::DriverOutsideBaffle { index: 0 })
        ));

        let mut non_unit = product();
        non_unit.drivers[0].normal = [0.0, 0.0, 2.0];
        assert!(matches!(
            non_unit.validate(),
            Err(ProductDesignError::InvalidDriver { index: 0, .. })
        ));
    }

    #[test]
    fn product_rejects_duplicate_driver_ids_before_artifact_or_drive_mapping() {
        let mut design = product();
        design.drivers[1].id = design.drivers[0].id.clone();

        assert!(matches!(
            design.validate(),
            Err(ProductDesignError::DuplicateDriverId { ref id }) if id == "woofer"
        ));
    }

    #[test]
    fn product_rejects_reused_crossover_branches_before_drive_mapping() {
        let mut design = product();
        design.drivers[1].crossover_branch = 0;

        assert!(matches!(
            design.validate(),
            Err(ProductDesignError::CrossoverBranchReuse {
                branch: 0,
                first_driver: 0,
                second_driver: 1,
            })
        ));
    }

    #[test]
    fn product_design_serde_admission_enforces_product_identity_contract() {
        let design = product();
        let encoded = serde_json::to_value(&design).expect("serialize valid product design");
        let decoded: ProductDesign =
            serde_json::from_value(encoded.clone()).expect("valid design must deserialize");
        assert_eq!(decoded, design);

        let mut duplicate = encoded;
        duplicate["drivers"][1]["id"] = serde_json::json!("woofer");
        let error = serde_json::from_value::<ProductDesign>(duplicate)
            .expect_err("serialized duplicate driver IDs must be rejected");
        assert!(error.to_string().contains("duplicate driver id"));

        let mut reused_branch = serde_json::to_value(&design).expect("serialize valid product");
        reused_branch["drivers"][1]["crossover_branch"] = serde_json::json!(0);
        let error = serde_json::from_value::<ProductDesign>(reused_branch)
            .expect_err("serialized reused crossover branches must be rejected");
        assert!(error.to_string().contains("must map one-to-one"));
    }

    #[test]
    fn product_design_artifact_round_trips_material_mapping_and_identity() {
        let mut design = product();
        design.cabinet.wall_material = CabinetMaterial::Plywood18mm;
        design.waveguide = Some(ProductWaveguideDesign {
            driver_id: "tweeter".into(),
            horizontal_spline: vec![[0.0, 0.04], [0.5, 0.03], [1.0, 0.025]],
            vertical_spline: vec![[0.0, 0.04], [0.5, 0.03], [1.0, 0.025]],
            depth_m: 0.02,
            surface_specific_impedance_pa_s_per_m: 1.7e6,
        });
        design.validate().expect("artifact fixture product valid");

        let artifact = design.to_artifact().expect("artifact construction");
        artifact.validate().expect("artifact validation");
        let encoded = artifact.canonical_json().expect("canonical product JSON");
        let decoded = ProductDesignArtifact::from_json(&encoded).expect("artifact replay");

        assert_eq!(decoded.design, design);
        assert_eq!(
            decoded.design.cabinet.wall_material,
            CabinetMaterial::Plywood18mm
        );
        assert_eq!(
            decoded
                .design
                .waveguide
                .as_ref()
                .expect("waveguide mapping")
                .surface_specific_impedance_pa_s_per_m,
            1.7e6
        );
        assert_eq!(decoded.canonical_json().expect("re-encode"), encoded);
        assert_eq!(artifact.content_sha256.len(), 64);
        assert!(
            artifact
                .content_sha256
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        );
    }

    #[test]
    fn product_design_artifact_rejects_schema_provenance_and_payload_tampering() {
        let design = product();
        let artifact = design.to_artifact().expect("artifact construction");

        let mut wrong_schema = artifact.clone();
        wrong_schema.schema_version += 1;
        assert!(matches!(
            wrong_schema.validate(),
            Err(ProductDesignError::InvalidProductArtifact(message))
                if message.contains("schema")
        ));

        let mut non_deterministic = artifact.clone();
        non_deterministic.provenance.deterministic = false;
        assert!(matches!(
            non_deterministic.validate(),
            Err(ProductDesignError::InvalidProductArtifact(message))
                if message.contains("deterministic")
        ));

        let mut tampered = serde_json::to_value(&artifact).expect("artifact JSON");
        tampered["design"]["drivers"][1]["id"] = serde_json::json!("woofer");
        let bytes = serde_json::to_vec(&tampered).expect("tampered JSON");
        assert!(matches!(
            ProductDesignArtifact::from_json(&bytes),
            Err(ProductDesignError::InvalidProductArtifact(_))
        ));
    }

    #[test]
    fn product_design_artifact_rejects_digest_tampering_and_accepts_provenance_reclassification() {
        let artifact = product().to_artifact().expect("artifact construction");
        let mut digest_tampered = serde_json::to_value(&artifact).expect("artifact JSON");
        digest_tampered["content_sha256"] = serde_json::json!("0".repeat(64));
        let bytes = serde_json::to_vec(&digest_tampered).expect("tampered digest JSON");
        assert!(matches!(
            ProductDesignArtifact::from_json(&bytes),
            Err(ProductDesignError::InvalidProductArtifact(message))
                if message.contains("digest")
        ));

        let content_sha256 = artifact.content_sha256.clone();
        let preview = artifact.with_provenance(ArtifactProvenance::native_preview());
        preview
            .validate()
            .expect("provenance reclassification preserves payload identity");
        assert_eq!(preview.content_sha256, content_sha256);
    }

    #[test]
    fn product_reference_artifact_binds_input_identity_and_replays() {
        let mut design = product();
        design.drivers.truncate(1);
        design.crossover.branches.truncate(1);
        let sweep = ProductReferenceSweep {
            speed_of_sound_m_per_s: 343.0,
            density_kg_per_m3: 1.204,
            interior_dofs: 1,
            aperture_elements: 1,
            driver_count: 1,
            provenance: ArtifactProvenance::native_golden(),
            points: vec![retained_mesh_level_point(
                120.0,
                100,
                200,
                "product-reference-artifact",
                1.0,
            )],
            rom_promotion: None,
            mesh_convergence: Vec::new(),
        };
        let artifact = ProductReferenceArtifact::from_design(&design, sweep)
            .expect("product reference artifact construction");
        artifact
            .validate_against_design(&design)
            .expect("bound product reference replay");
        let encoded = artifact.canonical_json().expect("canonical result JSON");
        let decoded = ProductReferenceArtifact::from_json(&encoded).expect("result replay");
        assert_eq!(decoded.canonical_json().expect("re-encode"), encoded);

        let mut tampered = serde_json::to_value(&artifact).expect("result JSON");
        tampered["product_content_sha256"] = serde_json::json!("0".repeat(64));
        let tampered_bytes = serde_json::to_vec(&tampered).expect("tampered result JSON");
        let tampered = ProductReferenceArtifact::from_json(&tampered_bytes)
            .expect("structurally valid but unbound result artifact");
        assert!(matches!(
            tampered.validate_against_design(&product()),
            Err(ProductDesignError::InvalidProductArtifact(message))
                if message.contains("input digest")
        ));

        design.drivers[0].position_m[0] += 0.001;
        assert!(matches!(
            artifact.validate_against_design(&design),
            Err(ProductDesignError::InvalidProductArtifact(message))
                if message.contains("digest")
        ));
    }

    #[test]
    fn port_driver_overlap_is_rejected() {
        let mut design = product();
        design.drivers[0].position_m = [0.0, 0.0, 0.0];
        assert!(matches!(
            design.validate(),
            Err(ProductDesignError::PortOverlapsDriver { index: 0 })
        ));
    }

    #[test]
    fn resolved_port_clearance_includes_tube_wall() {
        let mut design = product();
        design.drivers.truncate(1);
        design.crossover.branches.truncate(1);
        // This clears the legacy 25 mm bore by 10 mm, but not the 43 mm
        // physical tube radius supplied by the default resolved port.
        design.drivers[0].position_m = [0.06, 0.0, 0.0];
        assert!(matches!(
            design.validate(),
            Err(ProductDesignError::PortOverlapsDriver { index: 0 })
        ));

        design.cabinet.port.as_mut().expect("port").wall_thickness_m = None;
        assert!(design.validate().is_ok());
    }

    fn reference_point_for_mesh_test(
        frequency_hz: f64,
        driver_velocities_rms_m_per_s: Vec<Complex64>,
    ) -> ProductReferencePoint {
        ProductReferencePoint {
            frequency_hz,
            drive_velocity_rms_m_per_s: driver_velocities_rms_m_per_s
                .first()
                .copied()
                .unwrap_or_default(),
            driver_velocities_rms_m_per_s,
            aperture_volume_velocity_m3_per_s: Complex64::new(1.0, 0.0),
            mean_pressure_rms_pa: Complex64::new(1.0, 0.0),
            mean_abs_pressure_pa: 1.0,
            solver_iterations: 1,
            relative_residual: 0.0,
            monolithic_evidence: None,
            interface_evidence: None,
            thermal_evidence: None,
            structural_evidence: None,
        }
    }

    fn retained_mesh_level_point(
        frequency_hz: f64,
        mesh_node_count: usize,
        mesh_tetrahedron_count: usize,
        mesh_geometry_fingerprint: &str,
        scale: f64,
    ) -> ProductReferencePoint {
        let driver_velocity = Complex64::new(scale, 0.1 * scale);
        let aperture_flow = Complex64::new(2.0 * scale, -0.1 * scale);
        let mean_pressure = Complex64::new(4.0 * scale, 0.2 * scale);
        let point = ProductReferencePoint {
            frequency_hz,
            drive_velocity_rms_m_per_s: driver_velocity,
            driver_velocities_rms_m_per_s: vec![driver_velocity],
            aperture_volume_velocity_m3_per_s: aperture_flow,
            mean_pressure_rms_pa: mean_pressure,
            mean_abs_pressure_pa: mean_pressure.norm(),
            solver_iterations: 4,
            relative_residual: 1.0e-10,
            monolithic_evidence: Some(MonolithicProductReferenceEvidence {
                interior_relative_residual: 1.0e-10,
                exterior_relative_residual: 1.0e-10,
                mechanical_relative_residual: 1.0e-10,
                electrical_relative_residual: 1.0e-10,
                electrical_power_residual_w: 1.0e-10,
                mechanical_power_residual_w: 1.0e-10,
                port_volume_velocity_m3_per_s: aperture_flow,
                maximum_port_pressure_jump_pa: 1.0e-10,
                mesh_node_count,
                mesh_tetrahedron_count,
                mesh_geometry_fingerprint: mesh_geometry_fingerprint.into(),
                independent_validation: None,
            }),
            interface_evidence: Some(vec![InterfaceReferenceEvidence {
                interface_id: "driver-aperture-0".into(),
                relative_residual: 1.0e-10,
                power_residual_w: 1.0e-10,
                volume_flow_m3_per_s: aperture_flow,
                mesh_convergence_error: Some(0.0),
                port_flow_promotion: None,
            }]),
            thermal_evidence: None,
            structural_evidence: None,
        };
        point.validate().expect("synthetic mesh level point");
        point
    }

    fn retained_mesh_ladder() -> ProductMeshConvergenceEvidence {
        ProductMeshConvergenceEvidence::from_levels(vec![
            ProductMeshConvergenceLevel {
                label: "coarse".into(),
                characteristic_size_m: 0.004,
                point: retained_mesh_level_point(120.0, 100, 200, "mesh-coarse", 1.0),
            },
            ProductMeshConvergenceLevel {
                label: "medium".into(),
                characteristic_size_m: 0.003,
                point: retained_mesh_level_point(120.0, 180, 360, "mesh-medium", 1.005),
            },
            ProductMeshConvergenceLevel {
                label: "fine".into(),
                characteristic_size_m: 0.002,
                point: retained_mesh_level_point(120.0, 300, 600, "mesh-fine", 1.008),
            },
        ])
        .expect("three-level synthetic product mesh ladder")
    }

    fn accepted_port_flow_promotion(frequency_hz: f64) -> PortFlowPromotionEvidence {
        let level = |mesh_size_m: f64, dt_s: f64, scale: f64| PortFlowConvergenceLevel {
            mesh_size_m,
            dt_s,
            frequency_hz,
            port_flow: PortFlowPhasor {
                real: scale,
                imaginary: 0.2 * scale,
            },
            box_pressure: PortFlowPhasor {
                real: 4.0 * scale,
                imaginary: 0.5 * scale,
            },
            diaphragm_flow: PortFlowPhasor {
                real: 1.5 * scale,
                imaginary: 0.1 * scale,
            },
            max_linear_relative_residual: 1.0e-10,
            max_box_conservation_relative_residual: 1.0e-10,
            cap_continuity_relative_error: 1.0e-8,
            energy_algebraic_relative_closure: 1.0e-10,
            energy_physical_relative_closure: 1.0e-3,
            energy_interface_relative_residual: 1.0e-10,
        };

        PortFlowPromotionEvidence {
            schema_version: PORT_FLOW_PROMOTION_SCHEMA_VERSION,
            provenance: "product-port-flow-promotion-test.v1".into(),
            frequency_hz,
            spatial_levels: vec![
                level(0.004, 1.0e-5, 1.008),
                level(0.003, 1.0e-5, 1.004),
                level(0.002, 1.0e-5, 1.0),
            ],
            temporal_levels: vec![level(0.002, 2.0e-5, 1.004), level(0.002, 1.0e-5, 1.0)],
        }
    }

    fn runtime_port_flow_artifact(frequency_hz: f64) -> PortFlowRuntimePromotionArtifact {
        let evidence = accepted_port_flow_promotion(frequency_hz);
        PortFlowRuntimePromotionArtifact {
            plan: PortFlowRuntimePlan {
                spatial_mesh_sizes_m: vec![0.004, 0.003, 0.002],
                temporal_dt_s: vec![2.0e-5, 1.0e-5],
                frequency_hz,
                tail_cycles: 2,
                discarded_cycles: 2,
                box_volume_m3: 1.0e-3,
                density_kg_per_m3: 1.204,
                sound_speed_m_per_s: 343.0,
                provenance: evidence.provenance.clone(),
            },
            evidence,
        }
    }

    fn port_flow_attachment_sweep(frequencies_hz: &[f64]) -> ProductReferenceSweep {
        let points = frequencies_hz
            .iter()
            .enumerate()
            .map(|(index, frequency_hz)| {
                let mut point = retained_mesh_level_point(
                    *frequency_hz,
                    100 + index,
                    200 + index,
                    &format!("port-flow-product-{index}"),
                    1.0,
                );
                point
                    .interface_evidence
                    .as_mut()
                    .expect("retained point interface evidence")
                    .push(InterfaceReferenceEvidence {
                        interface_id: "port-mouth".into(),
                        relative_residual: 1.0e-10,
                        power_residual_w: 1.0e-10,
                        volume_flow_m3_per_s: Complex64::new(1.0, 0.0),
                        mesh_convergence_error: Some(1.0e-10),
                        port_flow_promotion: None,
                    });
                point
            })
            .collect();
        ProductReferenceSweep {
            speed_of_sound_m_per_s: 343.0,
            density_kg_per_m3: 1.204,
            interior_dofs: 300,
            aperture_elements: 12,
            points,
            driver_count: 1,
            provenance: ArtifactProvenance::native_golden(),
            rom_promotion: None,
            mesh_convergence: Vec::new(),
        }
    }

    #[test]
    fn attaches_runtime_port_flow_promotions_for_each_frequency() {
        let frequencies_hz = [80.0, 100.0, 125.0];
        let mut sweep = port_flow_attachment_sweep(&frequencies_hz);
        let artifacts = frequencies_hz
            .iter()
            .copied()
            .map(runtime_port_flow_artifact)
            .collect::<Vec<_>>();

        sweep
            .attach_port_flow_runtime_promotions(&artifacts)
            .expect("valid runtime promotion artifacts should attach");

        for (point, artifact) in sweep.points.iter().zip(&artifacts) {
            let evidence = point
                .interface_evidence
                .as_ref()
                .expect("attached point interface evidence")
                .iter()
                .find(|interface| interface.interface_id == "port-mouth")
                .expect("attached port-mouth evidence");
            assert_eq!(
                evidence.port_flow_promotion.as_ref(),
                Some(&artifact.evidence)
            );
        }
        sweep
            .validate()
            .expect("attached runtime promotion evidence should remain valid");
    }

    #[test]
    fn runtime_port_flow_attachment_rejects_frequency_mismatch_transactionally() {
        let frequencies_hz = [80.0, 100.0, 125.0];
        let mut sweep = port_flow_attachment_sweep(&frequencies_hz);
        let before = sweep.clone();
        let mut artifacts = frequencies_hz
            .iter()
            .copied()
            .map(runtime_port_flow_artifact)
            .collect::<Vec<_>>();
        artifacts[1] = runtime_port_flow_artifact(125.0);

        let error = sweep
            .attach_port_flow_runtime_promotions(&artifacts)
            .expect_err("frequency mismatch must be rejected");
        assert!(error.to_string().contains("frequency"), "{error}");
        assert_eq!(sweep, before, "failed attachment must not partially mutate");
    }

    #[test]
    fn runtime_port_flow_attachment_rejects_tampered_artifact_transactionally() {
        let frequencies_hz = [80.0, 100.0, 125.0];
        let mut sweep = port_flow_attachment_sweep(&frequencies_hz);
        let before = sweep.clone();
        let mut artifacts = frequencies_hz
            .iter()
            .copied()
            .map(runtime_port_flow_artifact)
            .collect::<Vec<_>>();
        artifacts[1].plan.spatial_mesh_sizes_m[2] = 0.001;

        let error = sweep
            .attach_port_flow_runtime_promotions(&artifacts)
            .expect_err("tampered runtime artifact must be rejected");
        assert!(
            error
                .to_string()
                .contains("invalid port-flow runtime promotion")
        );
        assert_eq!(sweep, before, "failed attachment must not partially mutate");
    }

    #[test]
    fn runtime_port_flow_attachment_requires_exactly_one_port_mouth() {
        let frequencies_hz = [80.0, 100.0, 125.0];
        let artifacts = frequencies_hz
            .iter()
            .copied()
            .map(runtime_port_flow_artifact)
            .collect::<Vec<_>>();
        let mut missing = port_flow_attachment_sweep(&frequencies_hz);
        missing.points[1]
            .interface_evidence
            .as_mut()
            .expect("interface evidence")
            .retain(|interface| interface.interface_id != "port-mouth");
        assert!(matches!(
            missing.attach_port_flow_runtime_promotions(&artifacts),
            Err(ProductDesignError::InvalidReferenceEvidence(message))
                if message.contains("exactly one port-mouth")
        ));

        let mut duplicate = port_flow_attachment_sweep(&frequencies_hz);
        duplicate.points[0]
            .interface_evidence
            .as_mut()
            .expect("interface evidence")
            .push(InterfaceReferenceEvidence {
                interface_id: "port-mouth".into(),
                relative_residual: 1.0e-10,
                power_residual_w: 1.0e-10,
                volume_flow_m3_per_s: Complex64::new(1.0, 0.0),
                mesh_convergence_error: Some(1.0e-10),
                port_flow_promotion: None,
            });
        let error = duplicate
            .attach_port_flow_runtime_promotions(&artifacts)
            .expect_err("duplicate port-mouth interfaces must be rejected");
        assert!(
            error.to_string().contains("duplicate interface ids"),
            "{error}"
        );
    }

    #[test]
    fn runtime_port_flow_attachment_rejects_existing_evidence() {
        let frequencies_hz = [80.0, 100.0, 125.0];
        let artifacts = frequencies_hz
            .iter()
            .copied()
            .map(runtime_port_flow_artifact)
            .collect::<Vec<_>>();
        let mut sweep = port_flow_attachment_sweep(&frequencies_hz);
        sweep.points[1]
            .interface_evidence
            .as_mut()
            .expect("interface evidence")
            .iter_mut()
            .find(|interface| interface.interface_id == "port-mouth")
            .expect("port-mouth interface")
            .port_flow_promotion = Some(accepted_port_flow_promotion(100.0));
        let before = sweep.clone();

        let error = sweep
            .attach_port_flow_runtime_promotions(&artifacts)
            .expect_err("existing promotion evidence must be rejected");
        assert!(error.to_string().contains("already contains"));
        assert_eq!(sweep, before, "failed attachment must not partially mutate");
    }

    #[test]
    fn reference_sweep_binds_to_canonical_product_coupling_mesh() {
        let mut design = ProductDesign {
            cabinet: CabinetDesign {
                port: None,
                ..CabinetDesign::default()
            },
            drivers: vec![driver("woofer", -0.06, 0.0, 0)],
            crossover: CrossoverNetwork {
                branches: vec![CrossoverBranch::default()],
            },
            waveguide: None,
            structural_acoustic: None,
        };
        design.drivers[0].radius_m = 0.045;
        design.drivers[0].position_m = [-0.06, 0.0, 0.0];
        let mut port = CabinetDesign::default().port.expect("default port");
        port.position_m = [0.06, 0.0];
        design.cabinet.port = Some(port);
        let mesh = ProductCouplingMesh::build(&design, ProductInteriorMeshConfig::default())
            .expect("canonical product mesh should build");
        let mesh_evidence = MonolithicProductReferenceEvidence {
            interior_relative_residual: 0.0,
            exterior_relative_residual: 0.0,
            mechanical_relative_residual: 0.0,
            electrical_relative_residual: 0.0,
            electrical_power_residual_w: 0.0,
            mechanical_power_residual_w: 0.0,
            port_volume_velocity_m3_per_s: Complex64::new(0.0, 0.0),
            maximum_port_pressure_jump_pa: 0.0,
            mesh_node_count: mesh.interior.diagnostics.node_count,
            mesh_tetrahedron_count: mesh.interior.diagnostics.tetrahedron_count,
            mesh_geometry_fingerprint: mesh.interior.geometry_fingerprint.clone(),
            independent_validation: None,
        };
        let point = ProductReferencePoint {
            frequency_hz: 90.0,
            drive_velocity_rms_m_per_s: Complex64::new(1.0e-3, 0.0),
            driver_velocities_rms_m_per_s: vec![Complex64::new(1.0e-3, 0.0)],
            aperture_volume_velocity_m3_per_s: Complex64::new(0.0, 0.0),
            mean_pressure_rms_pa: Complex64::new(1.0, 0.0),
            mean_abs_pressure_pa: 1.0,
            solver_iterations: 1,
            relative_residual: 0.0,
            monolithic_evidence: Some(mesh_evidence),
            interface_evidence: Some(vec![
                InterfaceReferenceEvidence {
                    interface_id: "driver-aperture-0".into(),
                    relative_residual: 0.0,
                    power_residual_w: 0.0,
                    volume_flow_m3_per_s: Complex64::new(0.0, 0.0),
                    mesh_convergence_error: Some(0.0),
                    port_flow_promotion: None,
                },
                InterfaceReferenceEvidence {
                    interface_id: "port-mouth".into(),
                    relative_residual: 0.0,
                    power_residual_w: 0.0,
                    volume_flow_m3_per_s: Complex64::new(0.0, 0.0),
                    mesh_convergence_error: Some(0.0),
                    port_flow_promotion: None,
                },
            ]),
            thermal_evidence: None,
            structural_evidence: None,
        };
        let sweep = ProductReferenceSweep {
            speed_of_sound_m_per_s: 343.0,
            density_kg_per_m3: 1.204,
            interior_dofs: mesh.interior.diagnostics.node_count,
            aperture_elements: 1,
            points: vec![point],
            driver_count: 1,
            provenance: ArtifactProvenance::native_golden(),
            rom_promotion: None,
            mesh_convergence: Vec::new(),
        };
        sweep
            .validate_against_product_coupling_mesh(&mesh)
            .expect("matching canonical product mesh should be accepted");

        let mut stale = sweep.clone();
        stale.points[0]
            .monolithic_evidence
            .as_mut()
            .expect("monolithic evidence")
            .mesh_geometry_fingerprint = "stale-mesh".into();
        assert!(matches!(
            stale.validate_against_product_coupling_mesh(&mesh),
            Err(ProductDesignError::InvalidReferenceEvidence(message))
                if message.contains("fingerprint")
        ));

        let mut missing_interface = sweep.clone();
        missing_interface.points[0]
            .interface_evidence
            .as_mut()
            .expect("interface evidence")
            .retain(|interface| interface.interface_id != "port-mouth");
        assert!(matches!(
            missing_interface.validate_against_product_coupling_mesh(&mesh),
            Err(ProductDesignError::InvalidReferenceEvidence(message))
                if message.contains("missing canonical interface")
        ));

        let mut extra_interface = sweep.clone();
        extra_interface.points[0]
            .interface_evidence
            .as_mut()
            .expect("interface evidence")
            .push(InterfaceReferenceEvidence {
                interface_id: "stale-interface".into(),
                relative_residual: 0.0,
                power_residual_w: 0.0,
                volume_flow_m3_per_s: Complex64::new(0.0, 0.0),
                mesh_convergence_error: Some(0.0),
                port_flow_promotion: None,
            });
        assert!(matches!(
            extra_interface.validate_against_product_coupling_mesh(&mesh),
            Err(ProductDesignError::InvalidReferenceEvidence(message))
                if message.contains("interface identities differ")
        ));

        let mut wrong_dofs = sweep;
        wrong_dofs.interior_dofs += 1;
        assert!(matches!(
            wrong_dofs.validate_against_product_coupling_mesh(&mesh),
            Err(ProductDesignError::InvalidReferenceEvidence(message))
                if message.contains("DOF count")
        ));
    }

    fn accepted_thermal_evidence(driver_ids: &[&str]) -> ProductThermalReferenceEvidence {
        let config = DriverThermalNetworkConfig::default();
        let source_power = ThermalNetworkPower {
            voice_coil_w: 2.0,
            magnet_w: 0.25,
            gap_w: 0.25,
            cabinet_w: 0.1,
        };
        let state = config
            .specification
            .solve_steady_state(source_power)
            .expect("thermal fixture should solve");
        ProductThermalReferenceEvidence {
            schema_version: PRODUCT_THERMAL_EVIDENCE_SCHEMA_VERSION,
            driver_ids: driver_ids.iter().map(|id| (*id).to_owned()).collect(),
            source_power_w: vec![source_power; driver_ids.len()],
            states: vec![state; driver_ids.len()],
            hot_coil_resistance_ohm: Some(vec![6.0; driver_ids.len()]),
            feedback_iterations: Some(2),
            feedback_relative_resistance_change: Some(1.0e-6),
        }
    }

    #[test]
    fn strict_thermal_promotion_requires_feedback_closure() {
        let mut evidence = accepted_thermal_evidence(&["woofer"]);
        evidence.hot_coil_resistance_ohm = None;
        evidence.feedback_iterations = None;
        evidence.feedback_relative_resistance_change = None;
        let gate = ProductThermalPromotionGate {
            require_feedback_closure: true,
            ..ProductThermalPromotionGate::default()
        };
        assert!(matches!(
            evidence.validate_for_driver_count(1, gate),
            Err(ProductDesignError::InvalidReferenceEvidence(message))
                if message.contains("missing electro-thermal feedback closure")
        ));

        evidence.hot_coil_resistance_ohm = Some(vec![6.0]);
        evidence.feedback_iterations = Some(2);
        evidence.feedback_relative_resistance_change = Some(1.0e-6);
        evidence
            .validate_for_driver_count(1, gate)
            .expect("feedback closure should satisfy strict thermal gate");
        assert!(
            MonolithicEvidenceGate::strict_thermoacoustic_product()
                .thermal
                .require_feedback_closure
        );
        assert!(
            MonolithicEvidenceGate::strict_full_product_scale()
                .thermal
                .require_feedback_closure
        );
    }

    fn accepted_structural_evidence(
        frequency_hz: f64,
        pressure_rms_pa: Complex64,
    ) -> ProductStructuralReferenceEvidence {
        let mut evidence = ProductStructuralReferenceEvidence::from_response(
            StructuralAcousticResponse {
                frequency_hz,
                pressure_rms_pa,
                displacement_rms_m: Complex64::new(2.0e-4, -1.0e-5),
                velocity_rms_m_per_s: Complex64::new(0.3, 0.02),
                stress_rms_pa: Complex64::new(120.0, 8.0),
                volume_velocity_rms_m3_per_s: Complex64::new(4.0e-5, 2.0e-6),
                absorbed_power_w: 0.25,
            },
            4,
            1.0e-10,
            1.0e-3,
        )
        .expect("structural fixture should validate");
        evidence.explicit_modal_relative_residual = Some(1.0e-10);
        evidence
    }

    #[test]
    fn strict_structural_promotion_requires_explicit_modal_residual() {
        let pressure = Complex64::new(1.0, 0.25);
        let mut evidence = accepted_structural_evidence(80.0, pressure);
        evidence.explicit_modal_relative_residual = None;
        let gate = ProductStructuralPromotionGate {
            require_explicit_modal_residual: true,
            ..ProductStructuralPromotionGate::default()
        };
        assert!(matches!(
            evidence.validate_for_point(80.0, pressure, gate),
            Err(ProductDesignError::InvalidReferenceEvidence(message))
                if message.contains("missing explicit modal residual")
        ));

        evidence.explicit_modal_relative_residual = Some(1.0e-10);
        evidence
            .validate_for_point(80.0, pressure, gate)
            .expect("explicit modal residual should satisfy strict gate");
        assert!(
            MonolithicEvidenceGate::strict_structural_product()
                .structural
                .require_explicit_modal_residual
        );
        assert!(
            MonolithicEvidenceGate::strict_full_product_scale()
                .structural
                .require_explicit_modal_residual
        );
    }

    #[test]
    fn product_structural_evidence_replays_production_feedback_model() {
        let feedback = StructuralAcousticFeedback {
            area_m2: 0.025,
            modes: vec![StructuralAcousticMode {
                natural_frequency_hz: 400.0,
                effective_mass_kg: 0.02,
                damping_ratio: 0.03,
                pressure_participation: 1.0,
                stress_per_displacement_pa_per_m: 30_000.0,
            }],
            max_iterations: 8,
            velocity_tolerance: 1.0e-7,
            under_relaxation: 0.8,
        };
        let frequency_hz = 80.0;
        let pressure_rms_pa = Complex64::new(1.0, 0.25);
        let expected = feedback
            .response_at(frequency_hz, pressure_rms_pa)
            .expect("valid production structural response");

        let evidence = ProductStructuralReferenceEvidence::from_feedback(
            &feedback,
            frequency_hz,
            pressure_rms_pa,
            3,
            1.0e-10,
            1.0e-3,
        )
        .expect("production response should become evidence");
        assert_eq!(evidence.response, expected);

        let invalid_feedback = StructuralAcousticFeedback {
            modes: Vec::new(),
            ..feedback
        };
        assert!(matches!(
            ProductStructuralReferenceEvidence::from_feedback(
                &invalid_feedback,
                frequency_hz,
                pressure_rms_pa,
                3,
                1.0e-10,
                1.0e-3,
            ),
            Err(ProductDesignError::InvalidStructuralFeedback(_))
        ));
    }

    #[test]
    fn product_structural_reference_evidence_requires_configured_feedback() {
        let mut design = product();
        let feedback = StructuralAcousticFeedback {
            area_m2: 0.025,
            modes: vec![StructuralAcousticMode {
                natural_frequency_hz: 400.0,
                effective_mass_kg: 0.02,
                damping_ratio: 0.03,
                pressure_participation: 1.0,
                stress_per_displacement_pa_per_m: 30_000.0,
            }],
            max_iterations: 8,
            velocity_tolerance: 1.0e-7,
            under_relaxation: 0.8,
        };
        design.structural_acoustic = Some(feedback);

        let evidence = design
            .structural_reference_evidence(80.0, Complex64::new(1.0, 0.25), 3, 1.0e-10, 1.0e-3)
            .expect("configured product feedback should produce evidence");
        assert_eq!(evidence.response.frequency_hz, 80.0);

        let unconfigured = product();
        assert!(matches!(
            unconfigured.structural_reference_evidence(
                80.0,
                Complex64::new(1.0, 0.25),
                3,
                1.0e-10,
                1.0e-3,
            ),
            Err(ProductDesignError::InvalidStructuralFeedback(message))
                if message.contains("no structural-acoustic feedback")
        ));
    }

    #[test]
    fn adjacent_mesh_convergence_error_compares_valid_observables() {
        let coarser = reference_point_for_mesh_test(
            90.0,
            vec![Complex64::new(1.0, 0.0), Complex64::new(0.5, 0.0)],
        );
        let mut finer = coarser.clone();
        finer.aperture_volume_velocity_m3_per_s = Complex64::new(1.01, 0.0);
        finer.mean_pressure_rms_pa = Complex64::new(0.99, 0.0);
        finer.driver_velocities_rms_m_per_s[1] = Complex64::new(0.505, 0.0);

        let error = adjacent_mesh_convergence_error(&coarser, &finer)
            .expect("valid adjacent mesh observables");
        assert!((error - 0.01).abs() < 1.0e-12);
    }

    #[test]
    fn adjacent_mesh_convergence_error_rejects_frequency_and_driver_mismatch() {
        let coarser = reference_point_for_mesh_test(90.0, vec![Complex64::new(1.0, 0.0)]);
        let wrong_frequency = reference_point_for_mesh_test(91.0, vec![Complex64::new(1.0, 0.0)]);
        assert!(matches!(
            adjacent_mesh_convergence_error(&coarser, &wrong_frequency),
            Err(ProductDesignError::InvalidReferenceEvidence(_))
        ));

        let wrong_driver_count = reference_point_for_mesh_test(
            90.0,
            vec![Complex64::new(1.0, 0.0), Complex64::new(1.0, 0.0)],
        );
        assert!(matches!(
            adjacent_mesh_convergence_error(&coarser, &wrong_driver_count),
            Err(ProductDesignError::InvalidReferenceEvidence(_))
        ));

        let mut with_interface = coarser.clone();
        with_interface.interface_evidence = Some(vec![InterfaceReferenceEvidence {
            interface_id: "port-mouth".into(),
            relative_residual: 0.0,
            power_residual_w: 0.0,
            volume_flow_m3_per_s: Complex64::new(1.0, 0.0),
            mesh_convergence_error: Some(0.0),
            port_flow_promotion: None,
        }]);
        let mut different_interface = with_interface.clone();
        different_interface
            .interface_evidence
            .as_mut()
            .expect("interface evidence")
            .first_mut()
            .expect("port interface")
            .interface_id = "driver-aperture-0".into();
        assert!(matches!(
            adjacent_mesh_convergence_error(&with_interface, &different_interface),
            Err(ProductDesignError::InvalidReferenceEvidence(_))
        ));
    }

    #[test]
    fn product_mesh_ladder_requires_three_refined_levels_and_recomputes_observables() {
        let ladder = retained_mesh_ladder();
        ladder.validate().expect("retained mesh ladder validates");
        assert_eq!(ladder.levels.len(), 3);
        assert_eq!(ladder.adjacent_observable_errors.len(), 2);
        assert!(ladder.maximum_adjacent_observable_error < 1.0e-2);

        let encoded = serde_json::to_value(&ladder).expect("serialize mesh ladder");
        let decoded: ProductMeshConvergenceEvidence =
            serde_json::from_value(encoded).expect("deserialize mesh ladder");
        decoded.validate().expect("round-tripped mesh ladder");
        assert_eq!(decoded, ladder);

        let too_short = ProductMeshConvergenceEvidence::from_levels(ladder.levels[..2].to_vec())
            .expect_err("two levels cannot be promoted");
        assert!(too_short.to_string().contains("at least three levels"));

        let mut malformed = ladder.clone();
        malformed.adjacent_observable_errors[0] = 0.0;
        assert!(matches!(
            malformed.validate(),
            Err(ProductDesignError::InvalidReferenceEvidence(message))
                if message.contains("does not match")
        ));
        let malformed_json = serde_json::to_value(&malformed).expect("serialize malformed ladder");
        assert!(serde_json::from_value::<ProductMeshConvergenceEvidence>(malformed_json).is_err());
    }

    #[test]
    fn strict_product_scale_promotion_requires_ladder_and_finest_identity() {
        let ladder = retained_mesh_ladder();
        let finest = ladder.finest_point().expect("fine point").clone();
        let mut sweep = ProductReferenceSweep {
            speed_of_sound_m_per_s: 343.0,
            density_kg_per_m3: 1.204,
            interior_dofs: 300,
            aperture_elements: 12,
            points: vec![finest],
            driver_count: 1,
            provenance: ArtifactProvenance::native_golden(),
            rom_promotion: None,
            mesh_convergence: vec![ladder],
        };
        sweep
            .validate_monolithic_product_evidence(MonolithicEvidenceGate::strict_product_scale())
            .expect("strict product-scale ladder should promote");
        let encoded = serde_json::to_string(&sweep).expect("serialize strict product sweep");
        let decoded: ProductReferenceSweep =
            serde_json::from_str(&encoded).expect("deserialize strict product sweep");
        decoded
            .validate_monolithic_product_evidence(MonolithicEvidenceGate::strict_product_scale())
            .expect("round-tripped strict product ladder should promote");

        let mut without_ladder = sweep.clone();
        without_ladder.mesh_convergence.clear();
        assert!(matches!(
            without_ladder
                .validate_monolithic_product_evidence(MonolithicEvidenceGate::strict_product_scale()),
            Err(ProductDesignError::InvalidReferenceEvidence(message))
                if message.contains("one mesh ladder per frequency")
        ));

        sweep.points[0].mean_pressure_rms_pa *= 1.1;
        assert!(matches!(
            sweep.validate_monolithic_product_evidence(MonolithicEvidenceGate::strict_product_scale()),
            Err(ProductDesignError::InvalidReferenceEvidence(message))
                if message.contains("differs from its retained finest")
        ));
    }

    #[test]
    fn strict_nonlinear_product_scale_requires_port_flow_evidence() {
        let mut ladder = retained_mesh_ladder();
        for level in &mut ladder.levels {
            level
                .point
                .interface_evidence
                .as_mut()
                .expect("mesh level interface evidence")
                .push(InterfaceReferenceEvidence {
                    interface_id: "port-mouth".into(),
                    relative_residual: 1.0e-10,
                    power_residual_w: 1.0e-10,
                    volume_flow_m3_per_s: Complex64::new(0.2, 0.0),
                    mesh_convergence_error: Some(1.0e-10),
                    port_flow_promotion: Some(accepted_port_flow_promotion(120.0)),
                });
        }
        ladder =
            ProductMeshConvergenceEvidence::from_levels(ladder.levels).expect("ported mesh ladder");
        let finest = ladder.finest_point().expect("fine point").clone();
        let mut sweep = ProductReferenceSweep {
            speed_of_sound_m_per_s: 343.0,
            density_kg_per_m3: 1.204,
            interior_dofs: 300,
            aperture_elements: 12,
            points: vec![finest],
            driver_count: 1,
            provenance: ArtifactProvenance::native_golden(),
            rom_promotion: None,
            mesh_convergence: vec![ladder],
        };

        sweep
            .validate_monolithic_product_evidence(
                MonolithicEvidenceGate::strict_nonlinear_product_scale(),
            )
            .expect("nonlinear product-scale evidence");

        sweep
            .points
            .first_mut()
            .expect("reference point")
            .interface_evidence
            .as_mut()
            .expect("interface evidence")
            .iter_mut()
            .find(|interface| interface.interface_id == "port-mouth")
            .expect("port-mouth interface")
            .port_flow_promotion = None;
        assert!(matches!(
            sweep.validate_monolithic_product_evidence(
                MonolithicEvidenceGate::strict_nonlinear_product_scale(),
            ),
            Err(ProductDesignError::InvalidReferenceEvidence(message))
                if message.contains("missing nonlinear port-flow promotion")
        ));
    }

    #[test]
    fn centred_single_driver_builds_distributed_fem_bem_reference() {
        let mut design = product();
        design.drivers.truncate(1);
        design.drivers[0].position_m = [0.0, 0.0, 0.0];
        design.crossover.branches.truncate(1);
        design.cabinet.port = None;
        let discretization = CylinderFixtureDiscretization {
            n_circumference: 12,
            n_box_depth: 3,
            n_cap_radial: 2,
            n_aperture_radial: 2,
            n_port_depth: 2,
        };
        let reference = design
            .build_fem_bem_reference(discretization, 343.0, 1.204)
            .unwrap();
        assert!(reference.num_dofs() > 0);
        assert!(reference.num_aperture_elements() > 0);
        let sweep = design
            .solve_fem_bem_reference_sweep(
                discretization,
                343.0,
                1.204,
                &[80.0],
                Complex64::new(2.0, 0.0),
                &sonium_solvers::LinearSolveConfig {
                    max_iterations: 500,
                    tolerance: 1.0e-7,
                    ..sonium_solvers::LinearSolveConfig::default()
                },
            )
            .unwrap();
        assert_eq!(sweep.points.len(), 1);
        assert_eq!(sweep.aperture_elements, reference.num_aperture_elements());
        assert!(sweep.points[0].mean_abs_pressure_pa.is_finite());
        assert!(sweep.points[0].relative_residual <= 1.0e-7);
        let panel_response = design.reference_cabinet_panel_response(&sweep, 2).unwrap();
        assert!(!panel_response.is_empty());
        assert!(panel_response.iter().all(|panel| {
            panel.peak_displacement_m.is_finite() && panel.peak_bending_stress_pa.is_finite()
        }));
    }

    #[test]
    fn centred_reference_retains_rom_promotion_evidence() {
        let mut design = product();
        design.drivers.truncate(1);
        design.drivers[0].position_m = [0.0, 0.0, 0.0];
        design.crossover.branches.truncate(1);
        design.cabinet.port = None;
        let discretization = CylinderFixtureDiscretization {
            n_circumference: 12,
            n_box_depth: 3,
            n_cap_radial: 2,
            n_aperture_radial: 2,
            n_port_depth: 2,
        };
        let sweep = design
            .solve_fem_bem_reference_sweep_with_rom(
                discretization,
                343.0,
                1.204,
                &[80.0],
                Complex64::new(1.0, 0.0),
                &sonium_solvers::LinearSolveConfig {
                    max_iterations: 500,
                    tolerance: 1.0e-7,
                    ..sonium_solvers::LinearSolveConfig::default()
                },
                RomPromotionPolicy {
                    max_relative_residual: 1.0,
                    max_full_order_solves: 1,
                    ..RomPromotionPolicy::default()
                },
            )
            .unwrap();
        let evidence = sweep.rom_promotion.as_ref().unwrap();
        assert_eq!(evidence.point_count, 1);
        assert!(evidence.promoted_count <= evidence.point_count);
        sweep.validate().unwrap();
        let encoded = serde_json::to_string(&sweep).unwrap();
        let decoded: ProductReferenceSweep = serde_json::from_str(&encoded).unwrap();
        decoded.validate().unwrap();
        assert_eq!(decoded.provenance, sweep.provenance);
        assert_eq!(decoded.points.len(), sweep.points.len());
        assert_eq!(
            decoded.rom_promotion.as_ref().unwrap().point_count,
            sweep.rom_promotion.as_ref().unwrap().point_count
        );
        assert!(
            (decoded.points[0].relative_residual - sweep.points[0].relative_residual).abs()
                < 1.0e-15
        );

        let mut malformed = serde_json::to_value(&sweep).unwrap();
        malformed["points"][0]["relative_residual"] = serde_json::json!(-1.0);
        let error = serde_json::from_value::<ProductReferenceSweep>(malformed)
            .expect_err("serialized reference evidence must be validated");
        assert!(error.to_string().contains("non-negative"));
    }

    #[test]
    fn product_design_variables_round_trip_and_move_drivers() {
        let design = product();
        let values: Vec<_> = design
            .design_variables()
            .into_iter()
            .map(|v| v.value)
            .collect();
        let restored = design.from_design_variables(&values).unwrap();
        assert_eq!(restored, design);
        let mut moved = values;
        let index = design.cabinet.design_variables().len();
        moved[index] += 0.01;
        let moved_design = design.from_design_variables(&moved).unwrap();
        assert!(
            (moved_design.drivers[0].position_m[0] - design.drivers[0].position_m[0] - 0.01).abs()
                < 1.0e-12
        );
    }

    #[test]
    fn product_bounded_taylor_probe_covers_interior_and_active_bound() {
        let design = product();
        let variables = design.design_variables();
        let gain_index = design.cabinet.design_variables().len() + 2 * design.drivers.len() + 1;
        let mut direction = vec![0.0; variables.len()];
        direction[gain_index] = 1.0;
        let values = variables
            .iter()
            .map(|variable| variable.value)
            .collect::<Vec<_>>();
        let evaluate_gain = |candidate: &ProductDesign| {
            Ok(candidate
                .drive_voltages_at(1_000.0, Complex64::new(1.0, 0.0))?
                .first()
                .copied()
                .ok_or(ProductDesignError::NoDrivers)?
                .norm())
        };
        let interior_error = design
            .bounded_taylor_error(&values, &direction, 1.0e-4, evaluate_gain)
            .unwrap();
        assert!(
            interior_error < 1.0e-8,
            "interior Taylor error {interior_error}"
        );

        let mut lower_values = values;
        lower_values[gain_index] = variables[gain_index].lower;
        let active_error = design
            .bounded_taylor_error(&lower_values, &direction, 1.0e-4, evaluate_gain)
            .unwrap();
        assert!(
            active_error < 1.0e-8,
            "active-bound Taylor error {active_error}"
        );

        let limits = crate::coupled_objectives::CoupledProductLimits {
            maximum_excursion_m: Some(1.0),
            ..Default::default()
        };
        let constraint_error = design
            .bounded_taylor_error(&lower_values, &direction, 1.0e-4, |candidate| {
                let velocity = candidate
                    .piston_velocities_at(1_000.0, Complex64::new(1.0, 0.0))?
                    .first()
                    .copied()
                    .ok_or(ProductDesignError::NoDrivers)?;
                let metrics = crate::coupled_objectives::CoupledProductMetrics::default()
                    .with_peak_excursion_m(velocity.norm());
                Ok(
                    crate::coupled_objectives::coupled_product_constraint_margins(
                        &metrics, &limits,
                    )[0],
                )
            })
            .unwrap();
        assert!(constraint_error.is_finite());
    }

    #[test]
    fn bounded_taylor_probe_rejects_invalid_inputs_and_nonfinite_objectives() {
        let design = product();
        let values = design
            .design_variables()
            .into_iter()
            .map(|variable| variable.value)
            .collect::<Vec<_>>();
        let direction = vec![1.0; values.len()];

        assert!(matches!(
            design.project_design_variables(&values[..values.len() - 1]),
            Err(ProductDesignError::DesignVariableCountMismatch { .. })
        ));

        let mut nonfinite = values.clone();
        nonfinite[0] = f64::NAN;
        assert!(matches!(
            design.project_design_variables(&nonfinite),
            Err(ProductDesignError::InvalidTaylorProbe(_))
        ));
        assert!(matches!(
            design.bounded_taylor_error(&values, &direction, 0.0, |_| Ok(0.0)),
            Err(ProductDesignError::InvalidTaylorProbe(_))
        ));
        assert!(matches!(
            design.bounded_taylor_error(&values, &direction[..direction.len() - 1], 1.0e-4, |_| {
                Ok(0.0)
            }),
            Err(ProductDesignError::InvalidTaylorProbe(_))
        ));
        assert!(matches!(
            design.bounded_taylor_error(&values, &vec![0.0; values.len()], 1.0e-4, |_| Ok(0.0)),
            Err(ProductDesignError::InvalidTaylorProbe(_))
        ));
        assert!(matches!(
            design.bounded_taylor_error(&values, &direction, 1.0e-4, |_| Ok(f64::NAN)),
            Err(ProductDesignError::InvalidTaylorProbe(_))
        ));
    }

    #[test]
    fn multi_driver_and_off_axis_products_cannot_use_axisymmetric_lowering() {
        let design = product();
        assert!(matches!(
            design.to_resolved_config(),
            Err(ProductDesignError::ResolvedModelRequiresSingleDriver)
        ));
        let mut centred = design;
        centred.drivers.truncate(1);
        centred.drivers[0].position_m = [0.0, 0.0, 0.0];
        // This axisymmetric lowering fixture models an air bore; omit the
        // physical tube wall so the intentionally close port stays clear of
        // the centred driver under product geometry validation.
        centred.cabinet.port.as_mut().unwrap().wall_thickness_m = None;
        centred.cabinet.port.as_mut().unwrap().position_m = [0.1, 0.0];
        centred.crossover.branches.truncate(1);
        centred.to_resolved_config().unwrap();
    }

    #[test]
    fn overlap_is_rejected_before_any_solver_call() {
        let mut design = product();
        design.drivers[1].position_m = design.drivers[0].position_m;
        assert!(matches!(
            design.validate(),
            Err(ProductDesignError::DriversOverlap { .. })
        ));
    }

    #[test]
    fn reference_sweep_rejects_empty_or_unvalidated_evidence() {
        let mut design = product();
        design.drivers.truncate(1);
        design.drivers[0].position_m = [0.0, 0.0, 0.0];
        design.crossover.branches.truncate(1);
        design.cabinet.port = None;
        let config = sonium_solvers::LinearSolveConfig::default();
        assert!(matches!(
            design.solve_fem_bem_reference_sweep(
                CylinderFixtureDiscretization::default(),
                343.0,
                1.204,
                &[],
                Complex64::new(1.0, 0.0),
                &config,
            ),
            Err(ProductDesignError::ReferenceSweepEmpty)
        ));

        let invalid = ProductReferenceSweep {
            speed_of_sound_m_per_s: 343.0,
            density_kg_per_m3: 1.204,
            interior_dofs: 1,
            aperture_elements: 1,
            points: vec![ProductReferencePoint {
                frequency_hz: 80.0,
                drive_velocity_rms_m_per_s: Complex64::new(0.0, 0.0),
                driver_velocities_rms_m_per_s: Vec::new(),
                aperture_volume_velocity_m3_per_s: Complex64::new(0.0, 0.0),
                mean_pressure_rms_pa: Complex64::new(1.0, 0.0),
                mean_abs_pressure_pa: 1.0,
                solver_iterations: 1,
                relative_residual: f64::NAN,
                monolithic_evidence: None,
                interface_evidence: None,
                thermal_evidence: None,
                structural_evidence: None,
            }],
            driver_count: 1,
            rom_promotion: None,
            provenance: ArtifactProvenance::native_golden(),
            mesh_convergence: Vec::new(),
        };
        assert!(matches!(
            invalid.validate(),
            Err(ProductDesignError::InvalidReferenceEvidence(_))
        ));
    }

    #[test]
    fn distributed_reference_accepts_multiple_planar_driver_profiles() {
        let mut design = product();
        design.cabinet.port = None;
        let discretization = CylinderFixtureDiscretization {
            n_circumference: 12,
            n_box_depth: 3,
            n_cap_radial: 2,
            n_aperture_radial: 2,
            n_port_depth: 2,
        };
        let coupling = design
            .build_fem_bem_reference(discretization, 343.0, 1.204)
            .unwrap();
        let profiles = design
            .drivers
            .iter()
            .map(|driver| DriverVelocityProfile {
                value: Complex64::new(1.0e-3, 0.0),
                center_x_m: driver.position_m[0],
                center_y_m: driver.position_m[1],
                radius_m: driver.radius_m,
            })
            .collect();
        let solution = coupling
            .solve_frequency(
                80.0,
                CouplingDrive::DriverVelocityProfiles(profiles),
                &sonium_solvers::LinearSolveConfig {
                    max_iterations: 500,
                    tolerance: 1.0e-7,
                    ..sonium_solvers::LinearSolveConfig::default()
                },
            )
            .unwrap();
        assert_eq!(design.drivers.len(), 2);
        assert!(
            solution
                .relative_residual
                .is_some_and(|value| value < 1.0e-7)
        );
        assert!(solution.mean_abs_pressure() > 0.0);
    }

    #[test]
    fn reference_sweep_rejects_driver_count_mismatch() {
        let mut sweep = ProductReferenceSweep {
            speed_of_sound_m_per_s: 343.0,
            density_kg_per_m3: 1.204,
            interior_dofs: 1,
            aperture_elements: 1,
            points: vec![ProductReferencePoint {
                frequency_hz: 80.0,
                drive_velocity_rms_m_per_s: Complex64::new(1.0e-3, 0.0),
                driver_velocities_rms_m_per_s: vec![
                    Complex64::new(1.0e-3, 0.0),
                    Complex64::new(1.0e-3, 0.0),
                ],
                aperture_volume_velocity_m3_per_s: Complex64::new(0.0, 0.0),
                mean_pressure_rms_pa: Complex64::new(1.0, 0.0),
                mean_abs_pressure_pa: 1.0,
                solver_iterations: 1,
                relative_residual: 1.0e-9,
                monolithic_evidence: None,
                interface_evidence: None,
                thermal_evidence: None,
                structural_evidence: None,
            }],
            driver_count: 1,
            provenance: ArtifactProvenance::native_golden(),
            rom_promotion: None,
            mesh_convergence: Vec::new(),
        };

        assert!(matches!(
            sweep.validate(),
            Err(ProductDesignError::InvalidReferenceEvidence(_))
        ));

        sweep.driver_count = 2;
        let mut missing_profile = sweep.points[0].clone();
        missing_profile.frequency_hz = 100.0;
        missing_profile.driver_velocities_rms_m_per_s.clear();
        sweep.points.push(missing_profile);
        assert!(matches!(
            sweep.validate(),
            Err(ProductDesignError::InvalidReferenceEvidence(message))
                if message.contains("must be present at every frequency")
        ));

        sweep.points[1].driver_velocities_rms_m_per_s =
            vec![Complex64::new(1.0e-3, 0.0), Complex64::new(1.0e-3, 0.0)];
        sweep.points[1].drive_velocity_rms_m_per_s = Complex64::new(2.0e-3, 0.0);
        assert!(matches!(
            sweep.validate(),
            Err(ProductDesignError::InvalidReferenceEvidence(message))
                if message.contains("differs from first distributed driver")
        ));
    }

    #[test]
    fn monolithic_evidence_gate_rejects_missing_and_unclosed_product_evidence() {
        let point = ProductReferencePoint {
            frequency_hz: 80.0,
            drive_velocity_rms_m_per_s: Complex64::new(1.0e-3, 0.0),
            driver_velocities_rms_m_per_s: vec![Complex64::new(1.0e-3, 0.0)],
            aperture_volume_velocity_m3_per_s: Complex64::default(),
            mean_pressure_rms_pa: Complex64::new(1.0, 0.0),
            mean_abs_pressure_pa: 1.0,
            solver_iterations: 1,
            relative_residual: 1.0e-9,
            monolithic_evidence: None,
            interface_evidence: None,
            thermal_evidence: None,
            structural_evidence: None,
        };
        let mut sweep = ProductReferenceSweep {
            speed_of_sound_m_per_s: 343.0,
            density_kg_per_m3: 1.204,
            interior_dofs: 4,
            aperture_elements: 1,
            points: vec![point],
            driver_count: 1,
            provenance: ArtifactProvenance::native_golden(),
            rom_promotion: None,
            mesh_convergence: Vec::new(),
        };
        assert!(
            sweep
                .validate_monolithic_product_evidence(MonolithicEvidenceGate::default())
                .is_err()
        );
        sweep.points[0].monolithic_evidence = Some(MonolithicProductReferenceEvidence {
            interior_relative_residual: 1.0e-10,
            exterior_relative_residual: 1.0e-10,
            mechanical_relative_residual: 1.0e-10,
            electrical_relative_residual: 1.0e-10,
            electrical_power_residual_w: 1.0e-10,
            mechanical_power_residual_w: 1.0e-10,
            port_volume_velocity_m3_per_s: Complex64::default(),
            maximum_port_pressure_jump_pa: 1.0e-10,
            mesh_node_count: 4,
            mesh_tetrahedron_count: 1,
            mesh_geometry_fingerprint: "product-interior-v1-test".into(),
            independent_validation: None,
        });
        sweep.points[0].interface_evidence = Some(vec![InterfaceReferenceEvidence {
            interface_id: "port-mouth".into(),
            relative_residual: 1.0e-10,
            power_residual_w: 1.0e-10,
            volume_flow_m3_per_s: Complex64::default(),
            mesh_convergence_error: None,
            port_flow_promotion: None,
        }]);
        assert!(matches!(
            sweep.validate_monolithic_product_evidence(MonolithicEvidenceGate::default()),
            Err(ProductDesignError::InvalidReferenceEvidence(message))
                if message.contains("missing mesh-convergence")
        ));
        sweep.points[0].interface_evidence = Some(vec![InterfaceReferenceEvidence {
            interface_id: "port-mouth".into(),
            relative_residual: 1.0e-10,
            power_residual_w: 1.0e-10,
            volume_flow_m3_per_s: Complex64::default(),
            mesh_convergence_error: Some(1.0e-10),
            port_flow_promotion: None,
        }]);
        sweep
            .validate_monolithic_product_evidence(MonolithicEvidenceGate::default())
            .expect("closed monolithic evidence");

        let strict_gate = MonolithicEvidenceGate::strict_nonlinear_product();
        assert!(matches!(
            sweep.validate_monolithic_product_evidence(strict_gate),
            Err(ProductDesignError::InvalidReferenceEvidence(message))
                if message.contains("missing nonlinear port-flow promotion")
        ));
        sweep.points[0]
            .interface_evidence
            .as_mut()
            .expect("interface evidence")[0]
            .port_flow_promotion = Some(accepted_port_flow_promotion(80.0));
        sweep
            .validate_monolithic_product_evidence(strict_gate)
            .expect("strict nonlinear product evidence");

        let strict_thermoacoustic_gate = MonolithicEvidenceGate::strict_thermoacoustic_product();
        assert!(matches!(
            sweep.validate_monolithic_product_evidence(strict_thermoacoustic_gate),
            Err(ProductDesignError::InvalidReferenceEvidence(message))
                if message.contains("missing thermal evidence")
        ));
        sweep.points[0].thermal_evidence = Some(accepted_thermal_evidence(&["woofer"]));
        sweep
            .validate_monolithic_product_evidence(strict_thermoacoustic_gate)
            .expect("strict thermoacoustic product evidence");

        let strict_structural_gate = MonolithicEvidenceGate::strict_structural_product();
        assert!(matches!(
            sweep.validate_monolithic_product_evidence(strict_structural_gate),
            Err(ProductDesignError::InvalidReferenceEvidence(message))
                if message.contains("missing structural evidence")
        ));
        sweep.points[0].structural_evidence =
            Some(accepted_structural_evidence(80.0, Complex64::new(1.0, 0.0)));
        sweep
            .validate_monolithic_product_evidence(strict_structural_gate)
            .expect("strict structural product evidence");

        let mut misplaced_port_evidence = sweep.clone();
        misplaced_port_evidence
            .points
            .first_mut()
            .expect("reference point")
            .interface_evidence
            .as_mut()
            .expect("interface evidence")[0]
            .interface_id = "driver-aperture-0".into();
        assert!(matches!(
            misplaced_port_evidence.validate(),
            Err(ProductDesignError::InvalidReferenceEvidence(message))
                if message.contains("must belong to port-mouth")
        ));

        let mut mismatched_interface = sweep.points[0].clone();
        mismatched_interface.frequency_hz = 100.0;
        mismatched_interface
            .interface_evidence
            .as_mut()
            .expect("interface evidence")[0]
            .port_flow_promotion = None;
        mismatched_interface
            .interface_evidence
            .as_mut()
            .expect("interface evidence")[0]
            .interface_id = "driver-aperture".into();
        mismatched_interface.structural_evidence = None;
        sweep.points.push(mismatched_interface);
        assert!(matches!(
            sweep.validate_monolithic_product_evidence(MonolithicEvidenceGate::default()),
            Err(ProductDesignError::InvalidReferenceEvidence(message))
                if message.contains("interface identities differ")
        ));

        sweep.points[1]
            .interface_evidence
            .as_mut()
            .expect("interface evidence")[0]
            .interface_id = "port-mouth".into();
        sweep.points[1]
            .monolithic_evidence
            .as_mut()
            .expect("evidence")
            .mesh_geometry_fingerprint = "different-mesh".into();
        assert!(matches!(
            sweep.validate_monolithic_product_evidence(MonolithicEvidenceGate::default()),
            Err(ProductDesignError::InvalidReferenceEvidence(message))
                if message.contains("mesh identity differs")
        ));

        sweep.points.pop();
        sweep.points[0]
            .monolithic_evidence
            .as_mut()
            .expect("evidence")
            .maximum_port_pressure_jump_pa = 1.0;
        assert!(
            sweep
                .validate_monolithic_product_evidence(MonolithicEvidenceGate::default())
                .is_err()
        );
    }

    #[test]
    fn product_thermal_evidence_rejects_schema_identity_and_temperature_failures() {
        let mut evidence = accepted_thermal_evidence(&["woofer"]);
        let gate = ProductThermalPromotionGate::default();
        evidence
            .validate_for_driver_ids(&["woofer".to_owned()], gate)
            .expect("accepted thermal evidence");

        evidence.schema_version += 1;
        assert!(
            evidence
                .validate_for_driver_ids(&["woofer".to_owned()], gate)
                .is_err()
        );
        evidence.schema_version = PRODUCT_THERMAL_EVIDENCE_SCHEMA_VERSION;
        evidence.driver_ids[0] = "other-driver".into();
        assert!(
            evidence
                .validate_for_driver_ids(&["woofer".to_owned()], gate)
                .is_err()
        );
        evidence.driver_ids[0] = "woofer".into();
        evidence.states[0].temperatures_k[0] = gate.max_temperature_k + 1.0;
        assert!(
            evidence
                .validate_for_driver_ids(&["woofer".to_owned()], gate)
                .is_err()
        );
        evidence.states[0].temperatures_k[0] = gate.max_temperature_k - 1.0;
        evidence.hot_coil_resistance_ohm = Some(vec![f64::NAN]);
        assert!(
            evidence
                .validate_for_driver_ids(&["woofer".to_owned()], gate)
                .is_err()
        );
        evidence.hot_coil_resistance_ohm = None;
        evidence.feedback_iterations = Some(1);
        assert!(
            evidence
                .validate_for_driver_ids(&["woofer".to_owned()], gate)
                .is_err()
        );
    }

    #[test]
    fn product_structural_evidence_rejects_tampering_and_unconverged_response() {
        let pressure = Complex64::new(1.0, 0.25);
        let mut evidence = accepted_structural_evidence(80.0, pressure);
        evidence
            .validate_for_point(80.0, pressure, ProductStructuralPromotionGate::default())
            .expect("accepted structural evidence");

        let encoded = serde_json::to_string(&evidence).expect("serialize structural evidence");
        let decoded: ProductStructuralReferenceEvidence =
            serde_json::from_str(&encoded).expect("deserialize structural evidence");
        assert_eq!(decoded, evidence);

        evidence.schema_version += 1;
        assert!(
            evidence
                .validate_for_point(80.0, pressure, ProductStructuralPromotionGate::default())
                .is_err()
        );
        evidence.schema_version = PRODUCT_STRUCTURAL_EVIDENCE_SCHEMA_VERSION;

        evidence.response.pressure_rms_pa *= 1.1;
        assert!(
            evidence
                .validate_for_point(80.0, pressure, ProductStructuralPromotionGate::default())
                .is_err()
        );
        evidence.response.pressure_rms_pa = pressure;

        evidence.relative_velocity_residual = 1.0;
        assert!(
            evidence
                .validate_for_point(80.0, pressure, ProductStructuralPromotionGate::default())
                .is_err()
        );
        evidence.relative_velocity_residual = 1.0e-10;
        evidence.response.absorbed_power_w = -1.0;
        assert!(
            evidence
                .validate_for_point(80.0, pressure, ProductStructuralPromotionGate::default())
                .is_err()
        );
    }

    #[test]
    fn monolithic_evidence_gate_separates_mesh_convergence_budget() {
        let point = ProductReferencePoint {
            frequency_hz: 80.0,
            drive_velocity_rms_m_per_s: Complex64::new(1.0e-3, 0.0),
            driver_velocities_rms_m_per_s: vec![Complex64::new(1.0e-3, 0.0)],
            aperture_volume_velocity_m3_per_s: Complex64::default(),
            mean_pressure_rms_pa: Complex64::new(1.0, 0.0),
            mean_abs_pressure_pa: 1.0,
            solver_iterations: 1,
            relative_residual: 1.0e-10,
            monolithic_evidence: Some(MonolithicProductReferenceEvidence {
                interior_relative_residual: 1.0e-10,
                exterior_relative_residual: 1.0e-10,
                mechanical_relative_residual: 1.0e-10,
                electrical_relative_residual: 1.0e-10,
                electrical_power_residual_w: 1.0e-10,
                mechanical_power_residual_w: 1.0e-10,
                port_volume_velocity_m3_per_s: Complex64::default(),
                maximum_port_pressure_jump_pa: 1.0e-10,
                mesh_node_count: 4,
                mesh_tetrahedron_count: 1,
                mesh_geometry_fingerprint: "mesh-v1".into(),
                independent_validation: None,
            }),
            interface_evidence: Some(vec![InterfaceReferenceEvidence {
                interface_id: "driver-aperture-0".into(),
                relative_residual: 1.0e-10,
                power_residual_w: 1.0e-10,
                volume_flow_m3_per_s: Complex64::default(),
                mesh_convergence_error: Some(1.1e-2),
                port_flow_promotion: None,
            }]),
            thermal_evidence: None,
            structural_evidence: None,
        };
        let sweep = ProductReferenceSweep {
            speed_of_sound_m_per_s: 343.0,
            density_kg_per_m3: 1.204,
            interior_dofs: 4,
            aperture_elements: 1,
            points: vec![point],
            driver_count: 1,
            provenance: ArtifactProvenance::native_golden(),
            rom_promotion: None,
            mesh_convergence: Vec::new(),
        };
        assert!(matches!(
            sweep.validate_monolithic_product_evidence(MonolithicEvidenceGate::default()),
            Err(ProductDesignError::InvalidReferenceEvidence(message))
                if message.contains("interface product reference evidence exceeds promotion gate")
        ));
        let mut gate = MonolithicEvidenceGate::default();
        gate.max_mesh_convergence_error = 1.1e-2;
        sweep
            .validate_monolithic_product_evidence(gate)
            .expect("mesh budget is independent from algebraic residual budget");
    }

    #[test]
    fn partitioned_promotion_requires_ported_monolithic_agreement_for_one_to_three_drivers() {
        for driver_count in 1..=3 {
            let points = [80.0, 160.0]
                .into_iter()
                .map(|frequency_hz| {
                    let velocities = (0..driver_count)
                        .map(|index| Complex64::new((index + 1) as f64 * 1.0e-3, 0.0))
                        .collect::<Vec<_>>();
                    let mut interfaces = vec![InterfaceReferenceEvidence {
                        interface_id: "port-mouth".into(),
                        relative_residual: 1.0e-10,
                        power_residual_w: 1.0e-10,
                        volume_flow_m3_per_s: Complex64::new(2.0e-5, 0.0),
                        mesh_convergence_error: Some(1.0e-10),
                        port_flow_promotion: None,
                    }];
                    interfaces.extend((0..driver_count).map(|index| InterfaceReferenceEvidence {
                        interface_id: format!("driver-aperture-{index}"),
                        relative_residual: 1.0e-10,
                        power_residual_w: 1.0e-10,
                        volume_flow_m3_per_s: velocities[index] * Complex64::new(1.0e-2, 0.0),
                        mesh_convergence_error: Some(1.0e-10),
                        port_flow_promotion: None,
                    }));
                    ProductReferencePoint {
                        frequency_hz,
                        drive_velocity_rms_m_per_s: velocities[0],
                        driver_velocities_rms_m_per_s: velocities,
                        aperture_volume_velocity_m3_per_s: Complex64::new(2.0e-5, 0.0),
                        mean_pressure_rms_pa: Complex64::new(1.0, 0.25),
                        mean_abs_pressure_pa: (1.0_f64 + 0.25_f64.powi(2)).sqrt(),
                        solver_iterations: 4,
                        relative_residual: 1.0e-10,
                        monolithic_evidence: Some(MonolithicProductReferenceEvidence {
                            interior_relative_residual: 1.0e-10,
                            exterior_relative_residual: 1.0e-10,
                            mechanical_relative_residual: 1.0e-10,
                            electrical_relative_residual: 1.0e-10,
                            electrical_power_residual_w: 1.0e-10,
                            mechanical_power_residual_w: 1.0e-10,
                            port_volume_velocity_m3_per_s: Complex64::new(2.0e-5, 0.0),
                            maximum_port_pressure_jump_pa: 1.0e-10,
                            mesh_node_count: 24,
                            mesh_tetrahedron_count: 48,
                            mesh_geometry_fingerprint: "ported-product-fixture-v1".into(),
                            independent_validation: None,
                        }),
                        interface_evidence: Some(interfaces),
                        thermal_evidence: None,
                        structural_evidence: None,
                    }
                })
                .collect::<Vec<_>>();
            let monolithic = ProductReferenceSweep {
                speed_of_sound_m_per_s: 343.0,
                density_kg_per_m3: 1.204,
                interior_dofs: 24,
                aperture_elements: 4,
                points,
                driver_count,
                provenance: ArtifactProvenance::native_golden(),
                rom_promotion: None,
                mesh_convergence: Vec::new(),
            };
            let mut partitioned = monolithic.clone();
            for point in &mut partitioned.points {
                point.monolithic_evidence = None;
            }
            partitioned
                .validate_partitioned_product_promotion_against(
                    &monolithic,
                    MonolithicEvidenceGate::default(),
                    PartitionedProductPromotionGate::default(),
                )
                .unwrap_or_else(|error| {
                    panic!("{driver_count}-driver ported promotion must pass: {error}")
                });

            partitioned.points[0].driver_velocities_rms_m_per_s[0] *= 2.0;
            partitioned.points[0].drive_velocity_rms_m_per_s =
                partitioned.points[0].driver_velocities_rms_m_per_s[0];
            assert!(matches!(
                    partitioned.validate_partitioned_product_promotion_against(
                        &monolithic,
                        MonolithicEvidenceGate::default(),
                        PartitionedProductPromotionGate::default(),
                    ),
                    Err(ProductDesignError::InvalidReferenceEvidence(message))
            if message.contains("driver velocity differs")
            ));
        }
    }

    #[test]
    fn strict_independent_product_validation_requires_recomputed_external_response() {
        let mut point = retained_mesh_level_point(120.0, 100, 200, "independent-product", 1.0);
        let mut sweep = ProductReferenceSweep {
            speed_of_sound_m_per_s: 343.0,
            density_kg_per_m3: 1.204,
            interior_dofs: 16,
            aperture_elements: 4,
            points: vec![point.clone()],
            driver_count: 1,
            provenance: ArtifactProvenance::native_golden(),
            rom_promotion: None,
            mesh_convergence: Vec::new(),
        };

        let strict_gate = MonolithicEvidenceGate::strict_independent_product();
        assert!(matches!(
            sweep.validate_monolithic_product_evidence(strict_gate),
            Err(ProductDesignError::InvalidReferenceEvidence(message))
                if message.contains("independent validation")
        ));

        let measured = point.mean_pressure_rms_pa * Complex64::new(1.01, 0.0);
        let relative_error = relative_complex_error(measured, point.mean_pressure_rms_pa, 1.0e-12);
        point
            .monolithic_evidence
            .as_mut()
            .expect("retained monolithic evidence")
            .independent_validation = Some(ProductIndependentValidationEvidence {
            schema_version: PRODUCT_INDEPENDENT_VALIDATION_SCHEMA_VERSION,
            metadata: IndependentMeasurementMetadata {
                dataset_id: "anechoic-product-2026-08".into(),
                source: "independent-lab".into(),
                acquisition_method: "calibrated-microphone".into(),
            },
            frequency_hz: point.frequency_hz,
            observable: ProductValidationObservable::MeanPressureRmsPa,
            measured_value: measured,
            relative_error,
        });
        sweep.points[0] = point.clone();
        sweep
            .validate_monolithic_product_evidence(strict_gate)
            .expect("valid independent response should promote");

        let encoded = serde_json::to_value(
            point
                .monolithic_evidence
                .as_ref()
                .expect("retained monolithic evidence")
                .independent_validation
                .as_ref()
                .expect("independent validation"),
        )
        .expect("serialize independent validation");
        let decoded: ProductIndependentValidationEvidence =
            serde_json::from_value(encoded).expect("deserialize independent validation");
        assert_eq!(
            decoded,
            point
                .monolithic_evidence
                .as_ref()
                .expect("retained monolithic evidence")
                .independent_validation
                .clone()
                .expect("independent validation")
        );

        let mut wrong_frequency = sweep.clone();
        wrong_frequency.points[0]
            .monolithic_evidence
            .as_mut()
            .expect("retained monolithic evidence")
            .independent_validation
            .as_mut()
            .expect("independent validation")
            .frequency_hz = 121.0;
        assert!(matches!(
            wrong_frequency.validate_monolithic_product_evidence(strict_gate),
            Err(ProductDesignError::InvalidReferenceEvidence(message))
                if message.contains("frequency differs")
        ));

        let mut wrong_error = sweep;
        wrong_error.points[0]
            .monolithic_evidence
            .as_mut()
            .expect("retained monolithic evidence")
            .independent_validation
            .as_mut()
            .expect("independent validation")
            .relative_error = 0.0;
        assert!(matches!(
            wrong_error.validate_monolithic_product_evidence(strict_gate),
            Err(ProductDesignError::InvalidReferenceEvidence(message))
                if message.contains("relative error does not match")
        ));
    }
}

#[cfg(test)]
mod active_crossover_tests {
    use super::*;
    #[test]
    fn linkwitz_riley_branches_have_flat_complex_sum_and_six_db_corner() {
        for (low, high) in [
            (
                CrossoverKind::LinkwitzRiley2LowPass,
                CrossoverKind::LinkwitzRiley2HighPass,
            ),
            (
                CrossoverKind::LinkwitzRiley4LowPass,
                CrossoverKind::LinkwitzRiley4HighPass,
            ),
        ] {
            let branch = |kind| CrossoverBranch {
                kind,
                corner_frequency_hz: 1000.0,
                gain: 1.0,
                delay_s: 0.0,
            };
            for f in [20.0, 100.0, 1000.0, 5000.0, 20000.0] {
                let sum =
                    branch(low).transfer_at(f).unwrap() + branch(high).transfer_at(f).unwrap();
                assert!((sum.norm() - 1.0).abs() < 1e-12);
            }
            assert!((branch(low).transfer_at(1000.0).unwrap().norm() - 0.5).abs() < 1e-12);
            assert!((branch(high).transfer_at(1000.0).unwrap().norm() - 0.5).abs() < 1e-12);
        }
    }
}
