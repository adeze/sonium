use crate::model::*;
use sonium_xem::io::{CellKind, MeshView, VtuWriter};
use std::fmt;
use std::path::Path;

const COLLISION_EPSILON_M: f64 = 1.0e-5;
const ROTOR_SUPPORT_CLEARANCE_M: f64 = 0.002;
const SHROUD_NOTCH_MARGIN: f64 = 1.0;

/// Invalid numerical input supplied to PC-fan geometry construction.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PcFanGeometryInputError {
    /// A named configuration dimension is NaN or infinite.
    NonFiniteDimension(&'static str),
    /// A flattened design variable is NaN or infinite.
    NonFiniteDesignVariable(usize),
    /// A named physical configuration dimension is outside its valid range.
    InvalidDimension {
        /// Configuration field name.
        field: &'static str,
        /// Required physical range.
        requirement: &'static str,
    },
    /// Two otherwise finite dimensions are physically contradictory.
    InvalidDimensionRelation {
        /// Required relation between configuration fields.
        requirement: &'static str,
    },
}

impl fmt::Display for PcFanGeometryInputError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NonFiniteDimension(name) => {
                write!(f, "PC-fan geometry dimension `{name}` must be finite")
            }
            Self::NonFiniteDesignVariable(index) => {
                write!(f, "PC-fan design variable {index} must be finite")
            }
            Self::InvalidDimension { field, requirement } => {
                write!(
                    f,
                    "PC-fan geometry dimension `{field}` must be {requirement}"
                )
            }
            Self::InvalidDimensionRelation { requirement } => {
                write!(f, "PC-fan geometry must satisfy {requirement}")
            }
        }
    }
}

impl std::error::Error for PcFanGeometryInputError {}

#[derive(Debug, Clone, Copy)]
pub(crate) struct ShroudNotch {
    pub(crate) angle_rad: f64,
    pub(crate) half_width_rad: f64,
    pub(crate) z_min: f64,
    pub(crate) z_max: f64,
}

pub(crate) fn shroud_strut_notches(
    struts: &[PcFanSupportStrut],
    shroud_radius_m: f64,
) -> Vec<ShroudNotch> {
    let r = shroud_radius_m.max(f64::EPSILON);
    struts
        .iter()
        .filter_map(|strut| {
            let dx = strut.start.x - strut.end.x;
            let dy = strut.start.y - strut.end.y;
            if dx.hypot(dy) < f64::EPSILON {
                return None;
            }
            let angle_rad = dy.atan2(dx);
            let half_width_rad = SHROUD_NOTCH_MARGIN * (0.5 * strut.width_m / r).atan();
            let z_max = strut.start.z;
            let z_min = z_max - strut.thickness_m;
            Some(ShroudNotch {
                angle_rad,
                half_width_rad,
                z_min,
                z_max,
            })
        })
        .collect()
}

pub(crate) fn shroud_in_notch(angle_rad: f64, z: f64, notches: &[ShroudNotch]) -> bool {
    notches.iter().any(|notch| {
        z >= notch.z_min
            && z <= notch.z_max
            && angular_distance(angle_rad, notch.angle_rad) < notch.half_width_rad
    })
}

/// Build the sorted list of azimuthal panel boundaries for a shroud
/// band/cap at the given `z`. Always includes the uniform circumferential
/// grid; when `z` lies inside a notch z-range, also splices in that
/// notch's two edge angles so cylinder panels stop exactly where the
/// strut starts (no rounding to the nearest segment).
fn shroud_panel_angles(base_segments: usize, z: f64, notches: &[ShroudNotch]) -> Vec<f64> {
    let mut angles: Vec<f64> = (0..=base_segments)
        .map(|i| std::f64::consts::TAU * i as f64 / base_segments as f64)
        .collect();
    for notch in notches {
        if z < notch.z_min - 1.0e-9 || z > notch.z_max + 1.0e-9 {
            continue;
        }
        for &delta in &[-notch.half_width_rad, notch.half_width_rad] {
            let edge = (notch.angle_rad + delta).rem_euclid(std::f64::consts::TAU);
            if !angles.iter().any(|&a| (a - edge).abs() < 1.0e-9) {
                angles.push(edge);
            }
        }
    }
    angles.sort_by(|a, b| a.total_cmp(b));
    angles
}

impl PcFanConfig {
    pub fn default_design_variables(&self) -> PcFanDesignVariables {
        PcFanDesignVariables::baseline_for(self)
    }

    pub fn build_geometry(&self, design: &PcFanDesignVariables) -> PcFanGeometry {
        self.try_build_geometry(design)
            .expect("invalid PC-fan geometry input")
    }

    /// Validate fixed physical dimensions independently of variable blade
    /// candidates. Candidate blade/shroud intersections intentionally remain
    /// evaluable by the optimizer so its collision penalty can steer them back
    /// into the feasible set.
    pub fn validate_geometry_config(&self) -> Result<(), PcFanGeometryInputError> {
        let dimensions = self.dimensions;
        for (field, value) in [
            ("frame_side_m", dimensions.frame_side_m),
            ("depth_m", dimensions.depth_m),
            ("hub_radius_m", dimensions.hub_radius_m),
            ("blade_tip_radius_m", dimensions.blade_tip_radius_m),
            ("shroud_inner_radius_m", dimensions.shroud_inner_radius_m),
            (
                "shroud_wall_thickness_m",
                dimensions.shroud_wall_thickness_m,
            ),
            ("frame_wall_thickness_m", dimensions.frame_wall_thickness_m),
            ("support_strut_width_m", dimensions.support_strut_width_m),
            (
                "support_strut_thickness_m",
                dimensions.support_strut_thickness_m,
            ),
            ("motor_mount_radius_m", dimensions.motor_mount_radius_m),
            ("frame_mount_inset_m", dimensions.frame_mount_inset_m),
            ("mount_pad_size_m", dimensions.mount_pad_size_m),
            (
                "rotor_strut_axial_gap_m",
                dimensions.rotor_strut_axial_gap_m,
            ),
            (
                "strut_wake_half_width_m",
                dimensions.strut_wake_half_width_m,
            ),
        ] {
            if !value.is_finite() || value <= 0.0 {
                return Err(PcFanGeometryInputError::InvalidDimension {
                    field,
                    requirement: "finite and positive",
                });
            }
        }
        if !dimensions.motor_mount_depth_m.is_finite() || dimensions.motor_mount_depth_m < 0.0 {
            return Err(PcFanGeometryInputError::InvalidDimension {
                field: "motor_mount_depth_m",
                requirement: "finite and non-negative",
            });
        }
        if !(0.0..=1.0).contains(&dimensions.strut_wake_centerline_deficit) {
            return Err(PcFanGeometryInputError::InvalidDimension {
                field: "strut_wake_centerline_deficit",
                requirement: "finite and within [0, 1]",
            });
        }
        if !(dimensions.hub_radius_m < dimensions.blade_tip_radius_m
            && dimensions.blade_tip_radius_m < dimensions.shroud_inner_radius_m)
        {
            return Err(PcFanGeometryInputError::InvalidDimensionRelation {
                requirement: "hub_radius_m < blade_tip_radius_m < shroud_inner_radius_m",
            });
        }
        let frame_inner_half_side =
            0.5 * dimensions.frame_side_m - dimensions.frame_wall_thickness_m;
        if frame_inner_half_side <= dimensions.shroud_inner_radius_m {
            return Err(PcFanGeometryInputError::InvalidDimensionRelation {
                requirement: "shroud_inner_radius_m must fit inside the frame opening",
            });
        }
        if dimensions.frame_mount_inset_m + 0.5 * dimensions.mount_pad_size_m
            >= 0.5 * dimensions.frame_side_m
        {
            return Err(PcFanGeometryInputError::InvalidDimensionRelation {
                requirement: "frame mounts must fit inside the frame side",
            });
        }
        Ok(())
    }

    /// Build geometry after rejecting non-finite dimensions and design values.
    pub fn try_build_geometry(
        &self,
        design: &PcFanDesignVariables,
    ) -> Result<PcFanGeometry, PcFanGeometryInputError> {
        let dimensions = self.dimensions;
        let dimension_values = [
            ("frame_side_m", dimensions.frame_side_m),
            ("depth_m", dimensions.depth_m),
            ("hub_radius_m", dimensions.hub_radius_m),
            ("blade_tip_radius_m", dimensions.blade_tip_radius_m),
            ("shroud_inner_radius_m", dimensions.shroud_inner_radius_m),
            (
                "shroud_wall_thickness_m",
                dimensions.shroud_wall_thickness_m,
            ),
            ("frame_wall_thickness_m", dimensions.frame_wall_thickness_m),
            ("support_strut_width_m", dimensions.support_strut_width_m),
            (
                "support_strut_thickness_m",
                dimensions.support_strut_thickness_m,
            ),
            ("motor_mount_radius_m", dimensions.motor_mount_radius_m),
            ("motor_mount_depth_m", dimensions.motor_mount_depth_m),
            ("frame_mount_inset_m", dimensions.frame_mount_inset_m),
            ("mount_pad_size_m", dimensions.mount_pad_size_m),
            (
                "rotor_strut_axial_gap_m",
                dimensions.rotor_strut_axial_gap_m,
            ),
            (
                "strut_wake_centerline_deficit",
                dimensions.strut_wake_centerline_deficit,
            ),
            (
                "strut_wake_half_width_m",
                dimensions.strut_wake_half_width_m,
            ),
        ];
        if let Some((name, _)) = dimension_values
            .iter()
            .find(|(_, value)| !value.is_finite())
        {
            return Err(PcFanGeometryInputError::NonFiniteDimension(name));
        }
        if let Some((index, _)) = design
            .to_vector()
            .as_slice()
            .iter()
            .enumerate()
            .find(|(_, value)| !value.is_finite())
        {
            return Err(PcFanGeometryInputError::NonFiniteDesignVariable(index));
        }

        self.validate_geometry_config()?;

        let hub_radius_m = design.effective_hub_radius_m(self);
        let tip_radius_m = design.effective_tip_radius_m(self);
        let minimum_radial_gap_m = if design.has_nonzero_radial_envelope_offsets() {
            RADIAL_ENVELOPE_MIN_GAP_M
        } else {
            0.0
        };
        if !(hub_radius_m > 0.0
            && hub_radius_m + minimum_radial_gap_m < tip_radius_m
            && tip_radius_m + minimum_radial_gap_m < dimensions.shroud_inner_radius_m)
        {
            return Err(PcFanGeometryInputError::InvalidDimensionRelation {
                requirement: "effective hub radius + minimum gap < effective blade tip radius + minimum gap < shroud_inner_radius_m",
            });
        }

        Ok(PcFanGeometry::from_config(self, design))
    }

    pub fn build_cfd_domain(&self, design: &PcFanDesignVariables) -> PcFanCfdDomain {
        self.build_geometry(design).build_cfd_domain(self)
    }
}

impl PcFanDesignVariables {
    pub fn baseline_for(config: &PcFanConfig) -> Self {
        // The hub sits forward of the support struts (it has clearance
        // baked into rotating_hub_z_bounds), so the centred-on-z=0
        // default leaves the blade trailing edge hanging below the hub
        // rear face. Anchor the baseline blade chord on the hub axial
        // centre instead, and clamp chord to a fraction of hub depth so
        // the root sits fully attached.
        let (hub_rear_z, hub_front_z) = rotating_hub_z_bounds(config.dimensions);
        let hub_center_z = 0.5 * (hub_rear_z + hub_front_z);
        let hub_depth = (hub_front_z - hub_rear_z).max(0.0);
        // Leave 20 % of hub depth as margin so the blade can rotate
        // freely without grazing the X struts that sit just below the
        // hub rear face (only ROTOR_SUPPORT_CLEARANCE_M = 2 mm separates
        // the hub rear from the strut top).
        let max_blade_chord_m = 0.80 * hub_depth;

        // Sensible curved-blade baseline. A flat plate with zero twist
        // and zero camber doesn't pump air, and the gradient signal is
        // tiny everywhere — leaving the optimiser stuck in a local
        // minimum where only the tip station moves. Start from a real
        // axial-fan profile:
        //   - twist: 30° at root → 12° at tip (steep root, shallow tip
        //     keeps angle of attack roughly constant across the span)
        //   - camber: ≈ 5 % of chord (mid-range for axial fans)
        //   - thickness: constant 1.2 mm (matches BladeSectionControl
        //     default)
        //   - chord: linearly tapered 0.88·depth → 0.56·depth (existing)
        const TWIST_DEG_ROOT: f64 = 30.0;
        const TWIST_DEG_TIP: f64 = 12.0;
        const CAMBER_FRACTION_OF_CHORD: f64 = 0.05;

        let mut blade_sections = [BladeSectionControl::default(); BLADE_SECTION_STATION_COUNT];
        for (i, section) in blade_sections.iter_mut().enumerate() {
            let eta = station_fraction(i, BLADE_SECTION_STATION_COUNT);
            let unconstrained_chord = 0.88 * config.dimensions.depth_m
                + eta * (0.56 * config.dimensions.depth_m - 0.88 * config.dimensions.depth_m);
            section.chord_m = unconstrained_chord.min(max_blade_chord_m);
            section.twist_deg = TWIST_DEG_ROOT + eta * (TWIST_DEG_TIP - TWIST_DEG_ROOT);
            section.camber_m = CAMBER_FRACTION_OF_CHORD * section.chord_m;
        }

        // Path baseline: backward-swept tip in plan view (the classic
        // axial-fan skew) plus a mild forward axial sweep. The
        // tangential offset is *negative* and grows quadratically toward
        // the tip — at azimuth 0 a negative tangential offset moves the
        // tip in `−Y`, i.e. opposite to the direction of rotation, which
        // makes the blade trail behind the rotor. Backward skew lengthens
        // the projected blade arc (more chord-perimeter exposed to the
        // flow) and is a known noise-reduction strategy. The optimiser
        // gets a non-flat starting point in *both* the path and section
        // spaces, and the wider tangential bound (`±30 mm`) leaves room
        // for it to grow the sweep further.
        //
        // The exact 5 mm at the tip is well inside the shroud-clearance
        // admissibility envelope (`sqrt(r_tip² + tang²) ≈ 55.2 mm` vs
        // shroud inner radius 57 mm — leaves ~1.8 mm of gap for the
        // unswept root margin to absorb the difference).
        const TIP_AXIAL_SWEEP_M: f64 = 0.003;
        const TIP_BACKWARD_SWEEP_M: f64 = 0.005;
        let mut blade_path = [BladePathControl::default(); BLADE_PATH_CONTROL_COUNT];
        for (i, control) in blade_path.iter_mut().enumerate() {
            let eta = station_fraction(i, BLADE_PATH_CONTROL_COUNT);
            control.axial_offset_m = hub_center_z + eta * eta * TIP_AXIAL_SWEEP_M;
            // Negative = backward in the rotation direction.
            control.tangential_offset_m = -eta * eta * TIP_BACKWARD_SWEEP_M;
        }

        Self {
            blade_path,
            blade_sections,
            hub_radius_offset_m: 0.0,
            tip_radius_offset_m: 0.0,
            shroud_radius_offsets_m: [0.0; SHROUD_CONTROL_COUNT],
        }
    }

    pub fn blade_section_at(&self, span_fraction: f64) -> BladeSectionControl {
        sample_spline(&self.blade_sections, span_fraction)
    }

    pub fn blade_path_at(&self, span_fraction: f64) -> BladePathControl {
        sample_spline(&self.blade_path, span_fraction)
    }

    pub fn shroud_radius_offset_at(&self, depth_fraction: f64) -> f64 {
        sample_spline(&self.shroud_radius_offsets_m, depth_fraction)
    }
}

impl PcFanGeometry {
    pub fn from_config(config: &PcFanConfig, design: &PcFanDesignVariables) -> Self {
        let dimensions = config.dimensions;
        let hub_radius_m = design.effective_hub_radius_m(config);
        let tip_radius_m = design.effective_tip_radius_m(config);
        let blades = (0..config.blade_count)
            .map(|blade_index| build_blade(config, design, blade_index, hub_radius_m, tip_radius_m))
            .collect();
        let shroud =
            PcFanShroud::from_design(dimensions, design, config.mesh.circumferential_segments);
        let frame_box = frame_box(dimensions, &shroud);
        let (hub_z0, hub_z1) = rotating_hub_z_bounds(dimensions);

        Self {
            hub: PcFanHub {
                radius_m: hub_radius_m,
                depth_m: hub_z1 - hub_z0,
                center_z_m: 0.5 * (hub_z0 + hub_z1),
                circumferential_segments: config.mesh.circumferential_segments,
            },
            blade_tip_radius_m: tip_radius_m,
            blades,
            shroud,
            frame_box,
            frame_mounts: fixed_frame_mounts(dimensions),
            support_struts: support_struts(dimensions, &frame_box),
            motor_mount: motor_mount(dimensions, config.mesh.circumferential_segments),
        }
    }

    pub fn validate_blade_intersections(&self) -> PcFanCollisionStatus {
        let report = blade_collision_report(self);

        PcFanCollisionStatus {
            is_valid: report.invalid_intersection_count == 0,
            root_merge_count: self.blades.len(),
            invalid_intersection_count: report.invalid_intersection_count,
            minimum_blade_gap_m: report.minimum_blade_gap_m,
        }
    }

    pub fn surface_panels(&self) -> Vec<PcFanSurfacePanel> {
        let blade_panel_count = self
            .blades
            .iter()
            .map(|blade| blade.panels.len())
            .sum::<usize>();
        let notches = shroud_strut_notches(&self.support_struts, self.shroud.minimum_radius_m());
        let mut panels = Vec::with_capacity(
            blade_panel_count
                + hub_panel_count(&self.hub)
                + shroud_panel_count(&self.shroud)
                + frame_box_panel_count()
                + self.support_struts.len() * support_strut_panel_count(),
        );

        panels.extend(frame_box_panels(&self.frame_box));
        panels.extend(hub_panels(&self.hub));
        for blade in &self.blades {
            panels.extend(blade.panels.iter().map(PcFanSurfacePanel::from));
        }
        panels.extend(shroud_panels(&self.shroud, &notches));
        panels.extend(self.support_struts.iter().flat_map(support_strut_panels));
        panels
    }

    pub fn build_cfd_domain(&self, config: &PcFanConfig) -> PcFanCfdDomain {
        build_structured_cfd_domain(config, self)
    }

    pub fn write_vtu(&self, path: impl AsRef<Path>) -> std::io::Result<()> {
        self.write_vtu_fields(path, None)
    }

    pub fn write_evaluation_vtu(
        &self,
        evaluation: &PcFanEvaluation,
        path: impl AsRef<Path>,
    ) -> std::io::Result<()> {
        self.write_vtu_fields(path, Some(evaluation))
    }

    fn write_vtu_fields(
        &self,
        path: impl AsRef<Path>,
        evaluation: Option<&PcFanEvaluation>,
    ) -> std::io::Result<()> {
        let panels = self.surface_panels();
        let view = PcFanVtuView { panels: &panels };
        let mut writer = VtuWriter::new(&view);

        let component: Vec<f64> = panels.iter().map(|p| p.kind.component_id()).collect();
        let blade_index: Vec<f64> = panels
            .iter()
            .map(|p| p.blade_index.map_or(-1.0, |i| i as f64))
            .collect();
        let span_fraction: Vec<f64> = panels.iter().map(|p| p.span_fraction).collect();
        let area: Vec<f64> = panels.iter().map(|p| p.area).collect();

        writer.add_cell_scalar("component", &component);
        writer.add_cell_scalar("blade_index", &blade_index);
        writer.add_cell_scalar("span_fraction", &span_fraction);
        writer.add_cell_scalar("area", &area);
        if let Some(evaluation) = evaluation {
            let axial_velocity = vec![evaluation.flow.axial_velocity_m_s; panels.len()];
            let pressure_rise = vec![evaluation.flow.pressure_rise_pa; panels.len()];
            let spl = vec![evaluation.acoustic_level_db; panels.len()];
            writer.add_cell_scalar("axial_velocity_m_s", &axial_velocity);
            writer.add_cell_scalar("pressure_rise_pa", &pressure_rise);
            writer.add_cell_scalar("overall_spl_db", &spl);
        }
        writer.write(path)
    }
}

impl PcFanCfdDomain {
    pub fn write_vtu(&self, path: impl AsRef<Path>) -> std::io::Result<()> {
        let view = PcFanCfdDomainVtuView { domain: self };
        let mut writer = VtuWriter::new(&view);

        let region: Vec<f64> = self
            .cells
            .iter()
            .map(|cell| cell.region.region_id())
            .collect();
        let volume: Vec<f64> = self.cells.iter().map(|cell| cell.volume_m3).collect();

        writer.add_cell_scalar("region", &region);
        writer.add_cell_scalar("volume_m3", &volume);
        writer.write(path)
    }

    pub fn write_boundary_vtu(&self, path: impl AsRef<Path>) -> std::io::Result<()> {
        let view = PcFanCfdBoundaryVtuView { domain: self };
        let mut writer = VtuWriter::new(&view);

        let boundary_kind: Vec<f64> = self
            .boundary_faces
            .iter()
            .map(|face| face.kind.boundary_id())
            .collect();
        let source_surface: Vec<f64> = self
            .boundary_faces
            .iter()
            .map(|face| face.source_surface.map_or(-1.0, |kind| kind.component_id()))
            .collect();
        let blade_index: Vec<f64> = self
            .boundary_faces
            .iter()
            .map(|face| face.blade_index.map_or(-1.0, |index| index as f64))
            .collect();
        let area: Vec<f64> = self.boundary_faces.iter().map(|face| face.area).collect();

        writer.add_cell_scalar("boundary_kind", &boundary_kind);
        writer.add_cell_scalar("source_surface", &source_surface);
        writer.add_cell_scalar("blade_index", &blade_index);
        writer.add_cell_scalar("area", &area);
        writer.write(path)
    }
}

impl PcFanCfdFlowSolution {
    pub fn write_vtu(&self, path: impl AsRef<Path>) -> std::io::Result<()> {
        let view = PcFanCfdFlowSolutionVtuView { solution: self };
        let mut writer = VtuWriter::new(&view);

        let vx: Vec<f64> = self
            .velocity_m_s
            .iter()
            .map(|velocity| velocity[0])
            .collect();
        let vy: Vec<f64> = self
            .velocity_m_s
            .iter()
            .map(|velocity| velocity[1])
            .collect();
        let vz: Vec<f64> = self
            .velocity_m_s
            .iter()
            .map(|velocity| velocity[2])
            .collect();
        let speed: Vec<f64> = self
            .velocity_m_s
            .iter()
            .map(|velocity| {
                (velocity[0] * velocity[0] + velocity[1] * velocity[1] + velocity[2] * velocity[2])
                    .sqrt()
            })
            .collect();
        let radius: Vec<f64> = self.nodes.iter().map(|node| node.x.hypot(node.y)).collect();
        let axial_position: Vec<f64> = self.nodes.iter().map(|node| node.z).collect();
        let (nearest_solid_kind, nearest_solid_distance) =
            nearest_solid_boundary_fields(&self.nodes, &self.domain.boundary_faces);
        let region: Vec<f64> = self
            .domain
            .cells
            .iter()
            .map(|cell| cell.region.region_id())
            .collect();
        let volume: Vec<f64> = self
            .domain
            .cells
            .iter()
            .map(|cell| cell.volume_m3)
            .collect();
        let iterations = vec![self.iterations as f64; self.domain.cells.len()];
        let residual = vec![self.residual; self.domain.cells.len()];
        let converged = vec![if self.converged { 1.0 } else { 0.0 }; self.domain.cells.len()];
        let conservation_relative_mass_imbalance =
            vec![self.conservation.relative_mass_imbalance; self.domain.cells.len()];
        let conservation_relative_momentum_imbalance =
            vec![self.conservation.relative_momentum_imbalance; self.domain.cells.len()];
        let conservation_relative_pressure_closure =
            vec![self.conservation.relative_pressure_closure; self.domain.cells.len()];
        let conservation_relative_energy_imbalance =
            vec![self.conservation.relative_energy_imbalance; self.domain.cells.len()];
        let conservation_pressure_power_w =
            vec![self.conservation.pressure_power_w; self.domain.cells.len()];
        let conservation_shaft_power_to_fluid_w =
            vec![self.conservation.shaft_power_to_fluid_w; self.domain.cells.len()];

        writer.add_point_vector("velocity_m_s", &vx, &vy, &vz);
        writer.add_point_scalar("pressure_pa", &self.pressure_pa);
        writer.add_point_scalar("speed_m_s", &speed);
        writer.add_point_scalar("radius_m", &radius);
        writer.add_point_scalar("axial_position_m", &axial_position);
        writer.add_point_scalar("nearest_solid_boundary_kind", &nearest_solid_kind);
        writer.add_point_scalar("nearest_solid_boundary_distance_m", &nearest_solid_distance);
        writer.add_cell_scalar("region", &region);
        writer.add_cell_scalar("volume_m3", &volume);
        writer.add_cell_scalar("picard_iterations", &iterations);
        writer.add_cell_scalar("picard_residual", &residual);
        writer.add_cell_scalar("picard_converged", &converged);
        writer.add_cell_scalar(
            "conservation_relative_mass_imbalance",
            &conservation_relative_mass_imbalance,
        );
        writer.add_cell_scalar(
            "conservation_relative_momentum_imbalance",
            &conservation_relative_momentum_imbalance,
        );
        writer.add_cell_scalar(
            "conservation_relative_pressure_closure",
            &conservation_relative_pressure_closure,
        );
        writer.add_cell_scalar(
            "conservation_relative_energy_imbalance",
            &conservation_relative_energy_imbalance,
        );
        writer.add_cell_scalar(
            "conservation_pressure_power_w",
            &conservation_pressure_power_w,
        );
        writer.add_cell_scalar(
            "conservation_shaft_power_to_fluid_w",
            &conservation_shaft_power_to_fluid_w,
        );
        writer.write(path)
    }
}

impl PcFanShroud {
    fn from_design(
        dimensions: PcFanDimensions,
        design: &PcFanDesignVariables,
        circumferential_segments: usize,
    ) -> Self {
        let max_inner_radius_m = (0.5 * dimensions.frame_side_m
            - dimensions.frame_wall_thickness_m
            - dimensions.shroud_wall_thickness_m)
            .max(0.0);
        let mut radius_controls_m = [dimensions.shroud_inner_radius_m; SHROUD_CONTROL_COUNT];
        for (i, radius) in radius_controls_m.iter_mut().enumerate() {
            let depth_fraction = station_fraction(i, SHROUD_CONTROL_COUNT);
            *radius =
                (*radius + design.shroud_radius_offset_at(depth_fraction)).min(max_inner_radius_m);
        }

        Self {
            inner_radius_m: radius_controls_m
                .iter()
                .copied()
                .fold(f64::INFINITY, f64::min),
            outer_radius_m: radius_controls_m.iter().copied().fold(0.0, f64::max)
                + dimensions.shroud_wall_thickness_m,
            wall_thickness_m: dimensions.shroud_wall_thickness_m,
            radius_controls_m,
            depth_m: dimensions.depth_m,
            axial_segments: SHROUD_CONTROL_COUNT - 1,
            circumferential_segments,
        }
    }

    pub fn radius_at_depth_fraction(&self, depth_fraction: f64) -> f64 {
        sample_spline(&self.radius_controls_m, depth_fraction)
    }

    pub(crate) fn radius_slope_at_depth_fraction(&self, depth_fraction: f64) -> f64 {
        sample_spline_derivative(&self.radius_controls_m, depth_fraction)
    }

    pub fn outer_radius_at_depth_fraction(&self, depth_fraction: f64) -> f64 {
        self.radius_at_depth_fraction(depth_fraction) + self.wall_thickness_m
    }

    pub fn minimum_radius_m(&self) -> f64 {
        let samples = self.axial_segments.max(1) * 4;
        (0..=samples)
            .map(|i| self.radius_at_depth_fraction(i as f64 / samples as f64))
            .fold(f64::INFINITY, f64::min)
    }
}

impl PcFanSurfaceKind {
    fn component_id(self) -> f64 {
        match self {
            Self::Hub => 0.0,
            Self::Blade => 1.0,
            Self::Shroud => 2.0,
            Self::FrameBox => 3.0,
            Self::FrameMount => 4.0,
            Self::SupportStrut => 5.0,
            Self::MotorMount => 6.0,
        }
    }
}

impl PcFanCfdCellRegion {
    fn region_id(self) -> f64 {
        match self {
            Self::InletExtension => 0.0,
            Self::FanBox => 1.0,
            Self::OutletExtension => 2.0,
        }
    }
}

impl PcFanCfdBoundaryKind {
    fn boundary_id(self) -> f64 {
        match self {
            Self::Inlet => 0.0,
            Self::Outlet => 1.0,
            Self::OuterWall => 2.0,
            Self::FrameWall => 3.0,
            Self::ShroudWall => 4.0,
            Self::SupportWall => 5.0,
            Self::MotorWall => 6.0,
            Self::RotatingHub => 7.0,
            Self::RotatingBlade => 8.0,
        }
    }
}

impl From<&PcFanBladePanel> for PcFanSurfacePanel {
    fn from(panel: &PcFanBladePanel) -> Self {
        Self {
            vertices: panel.vertices,
            kind: PcFanSurfaceKind::Blade,
            blade_index: Some(panel.blade_index),
            span_fraction: panel.span_fraction,
            area: panel.area,
        }
    }
}

fn build_structured_cfd_domain(config: &PcFanConfig, geometry: &PcFanGeometry) -> PcFanCfdDomain {
    let nx = config.cfd.cells_x.max(1);
    let ny = config.cfd.cells_y.max(1);
    let nz = config.cfd.cells_z.max(1);
    let half_side_m = 0.5 * geometry.frame_box.outer_side_m + config.cfd.side_clearance_m.max(0.0);
    let fan_z0 = -0.5 * geometry.frame_box.depth_m;
    let fan_z1 = 0.5 * geometry.frame_box.depth_m;
    let z0 = fan_z0 - config.cfd.inlet_extension_m.max(0.0);
    let z1 = fan_z1 + config.cfd.outlet_extension_m.max(0.0);
    let min_corner = Point3::new(-half_side_m, -half_side_m, z0);
    let max_corner = Point3::new(half_side_m, half_side_m, z1);
    let dx = (max_corner.x - min_corner.x) / nx as f64;
    let dy = (max_corner.y - min_corner.y) / ny as f64;
    let dz = (max_corner.z - min_corner.z) / nz as f64;
    let cell_volume_m3 = dx * dy * dz;

    let mut nodes = Vec::with_capacity((nx + 1) * (ny + 1) * (nz + 1));
    for iz in 0..=nz {
        for iy in 0..=ny {
            for ix in 0..=nx {
                nodes.push(Point3::new(
                    min_corner.x + dx * ix as f64,
                    min_corner.y + dy * iy as f64,
                    min_corner.z + dz * iz as f64,
                ));
            }
        }
    }

    let mut cells = Vec::with_capacity(nx * ny * nz);
    for iz in 0..nz {
        for iy in 0..ny {
            for ix in 0..nx {
                let center_z = min_corner.z + dz * (iz as f64 + 0.5);
                let region = if center_z < fan_z0 {
                    PcFanCfdCellRegion::InletExtension
                } else if center_z > fan_z1 {
                    PcFanCfdCellRegion::OutletExtension
                } else {
                    PcFanCfdCellRegion::FanBox
                };
                cells.push(PcFanCfdCell {
                    node_indices: [
                        structured_node_index(ix, iy, iz, nx, ny),
                        structured_node_index(ix + 1, iy, iz, nx, ny),
                        structured_node_index(ix + 1, iy + 1, iz, nx, ny),
                        structured_node_index(ix, iy + 1, iz, nx, ny),
                        structured_node_index(ix, iy, iz + 1, nx, ny),
                        structured_node_index(ix + 1, iy, iz + 1, nx, ny),
                        structured_node_index(ix + 1, iy + 1, iz + 1, nx, ny),
                        structured_node_index(ix, iy + 1, iz + 1, nx, ny),
                    ],
                    region,
                    volume_m3: cell_volume_m3,
                });
            }
        }
    }

    let mut boundary_faces = structured_domain_boundary_faces(&nodes, nx, ny, nz);
    boundary_faces.extend(
        geometry
            .surface_panels()
            .into_iter()
            .map(cfd_solid_boundary_face),
    );

    let inlet_area_m2 = (2.0 * half_side_m).powi(2);
    PcFanCfdDomain {
        nodes,
        cells,
        boundary_faces,
        min_corner,
        max_corner,
        inlet_area_m2,
        outlet_area_m2: inlet_area_m2,
        volume_m3: inlet_area_m2 * (z1 - z0),
    }
}

fn structured_node_index(ix: usize, iy: usize, iz: usize, nx: usize, ny: usize) -> usize {
    iz * (ny + 1) * (nx + 1) + iy * (nx + 1) + ix
}

fn structured_domain_boundary_faces(
    nodes: &[Point3],
    nx: usize,
    ny: usize,
    nz: usize,
) -> Vec<PcFanCfdBoundaryFace> {
    let mut faces = Vec::with_capacity(2 * (nx * ny + nx * nz + ny * nz));

    for iy in 0..ny {
        for ix in 0..nx {
            faces.push(cfd_boundary_face_from_nodes(
                nodes,
                [
                    structured_node_index(ix, iy, 0, nx, ny),
                    structured_node_index(ix + 1, iy, 0, nx, ny),
                    structured_node_index(ix + 1, iy + 1, 0, nx, ny),
                    structured_node_index(ix, iy + 1, 0, nx, ny),
                ],
                PcFanCfdBoundaryKind::Inlet,
            ));
            faces.push(cfd_boundary_face_from_nodes(
                nodes,
                [
                    structured_node_index(ix, iy, nz, nx, ny),
                    structured_node_index(ix, iy + 1, nz, nx, ny),
                    structured_node_index(ix + 1, iy + 1, nz, nx, ny),
                    structured_node_index(ix + 1, iy, nz, nx, ny),
                ],
                PcFanCfdBoundaryKind::Outlet,
            ));
        }
    }

    for iz in 0..nz {
        for iy in 0..ny {
            faces.push(cfd_boundary_face_from_nodes(
                nodes,
                [
                    structured_node_index(0, iy, iz, nx, ny),
                    structured_node_index(0, iy + 1, iz, nx, ny),
                    structured_node_index(0, iy + 1, iz + 1, nx, ny),
                    structured_node_index(0, iy, iz + 1, nx, ny),
                ],
                PcFanCfdBoundaryKind::OuterWall,
            ));
            faces.push(cfd_boundary_face_from_nodes(
                nodes,
                [
                    structured_node_index(nx, iy, iz, nx, ny),
                    structured_node_index(nx, iy, iz + 1, nx, ny),
                    structured_node_index(nx, iy + 1, iz + 1, nx, ny),
                    structured_node_index(nx, iy + 1, iz, nx, ny),
                ],
                PcFanCfdBoundaryKind::OuterWall,
            ));
        }
        for ix in 0..nx {
            faces.push(cfd_boundary_face_from_nodes(
                nodes,
                [
                    structured_node_index(ix, 0, iz, nx, ny),
                    structured_node_index(ix, 0, iz + 1, nx, ny),
                    structured_node_index(ix + 1, 0, iz + 1, nx, ny),
                    structured_node_index(ix + 1, 0, iz, nx, ny),
                ],
                PcFanCfdBoundaryKind::OuterWall,
            ));
            faces.push(cfd_boundary_face_from_nodes(
                nodes,
                [
                    structured_node_index(ix, ny, iz, nx, ny),
                    structured_node_index(ix + 1, ny, iz, nx, ny),
                    structured_node_index(ix + 1, ny, iz + 1, nx, ny),
                    structured_node_index(ix, ny, iz + 1, nx, ny),
                ],
                PcFanCfdBoundaryKind::OuterWall,
            ));
        }
    }

    faces
}

fn cfd_boundary_face_from_nodes(
    nodes: &[Point3],
    indices: [usize; 4],
    kind: PcFanCfdBoundaryKind,
) -> PcFanCfdBoundaryFace {
    let vertices = [
        nodes[indices[0]],
        nodes[indices[1]],
        nodes[indices[2]],
        nodes[indices[3]],
    ];
    cfd_boundary_face(vertices, kind, None, None)
}

fn cfd_solid_boundary_face(panel: PcFanSurfacePanel) -> PcFanCfdBoundaryFace {
    cfd_boundary_face(
        panel.vertices,
        cfd_boundary_kind_for_surface(panel.kind),
        Some(panel.kind),
        panel.blade_index,
    )
}

fn cfd_boundary_kind_for_surface(kind: PcFanSurfaceKind) -> PcFanCfdBoundaryKind {
    match kind {
        PcFanSurfaceKind::Hub => PcFanCfdBoundaryKind::RotatingHub,
        PcFanSurfaceKind::Blade => PcFanCfdBoundaryKind::RotatingBlade,
        PcFanSurfaceKind::Shroud => PcFanCfdBoundaryKind::ShroudWall,
        PcFanSurfaceKind::FrameBox | PcFanSurfaceKind::FrameMount => {
            PcFanCfdBoundaryKind::FrameWall
        }
        PcFanSurfaceKind::SupportStrut => PcFanCfdBoundaryKind::SupportWall,
        PcFanSurfaceKind::MotorMount => PcFanCfdBoundaryKind::MotorWall,
    }
}

fn cfd_boundary_face(
    vertices: [Point3; 4],
    kind: PcFanCfdBoundaryKind,
    source_surface: Option<PcFanSurfaceKind>,
    blade_index: Option<usize>,
) -> PcFanCfdBoundaryFace {
    PcFanCfdBoundaryFace {
        vertices,
        kind,
        source_surface,
        blade_index,
        area: quad_area(vertices),
    }
}

fn build_blade(
    config: &PcFanConfig,
    design: &PcFanDesignVariables,
    blade_index: usize,
    hub_radius_m: f64,
    tip_radius_m: f64,
) -> PcFanBlade {
    let azimuth_rad = std::f64::consts::TAU * blade_index as f64 / config.blade_count as f64;
    let mut panels = Vec::with_capacity(config.mesh.radial_segments * config.mesh.chord_segments);

    for ir in 0..config.mesh.radial_segments {
        let r0 = ir as f64 / config.mesh.radial_segments as f64;
        let r1 = (ir + 1) as f64 / config.mesh.radial_segments as f64;
        for ic in 0..config.mesh.chord_segments {
            let c0 = ic as f64 / config.mesh.chord_segments as f64;
            let c1 = (ic + 1) as f64 / config.mesh.chord_segments as f64;
            let vertices = [
                blade_point(design, azimuth_rad, r0, c0, hub_radius_m, tip_radius_m),
                blade_point(design, azimuth_rad, r1, c0, hub_radius_m, tip_radius_m),
                blade_point(design, azimuth_rad, r1, c1, hub_radius_m, tip_radius_m),
                blade_point(design, azimuth_rad, r0, c1, hub_radius_m, tip_radius_m),
            ];
            let span_fraction = 0.5 * (r0 + r1);
            let section = design.blade_section_at(span_fraction);
            panels.push(PcFanBladePanel {
                vertices,
                blade_index,
                span_fraction,
                chord_fraction: 0.5 * (c0 + c1),
                thickness_m: section.thickness_m.max(0.0),
                area: quad_area(vertices),
            });
        }
    }

    PcFanBlade {
        index: blade_index,
        azimuth_rad,
        panels,
    }
}

fn blade_point(
    design: &PcFanDesignVariables,
    azimuth_rad: f64,
    span_fraction: f64,
    chord_fraction: f64,
    hub_radius_m: f64,
    tip_radius_m: f64,
) -> Point3 {
    let radius = hub_radius_m + span_fraction * (tip_radius_m - hub_radius_m);
    let section = design.blade_section_at(span_fraction);
    let path = design.blade_path_at(span_fraction);
    let chord_centered = chord_fraction - 0.5;
    let camber_shape = 1.0 - (2.0 * chord_centered).powi(2);
    let twist_rad = section.twist_deg.to_radians();
    let local_chord = chord_centered * section.chord_m;
    let tangential_offset = path.tangential_offset_m
        + local_chord * twist_rad.sin()
        + section.camber_m * camber_shape.max(0.0);
    let axial_offset = path.axial_offset_m + local_chord * twist_rad.cos();

    polar_point(radius, azimuth_rad, tangential_offset, axial_offset)
}

fn polar_point(radius: f64, azimuth_rad: f64, tangential_offset: f64, z: f64) -> Point3 {
    let (sin_theta, cos_theta) = azimuth_rad.sin_cos();
    Point3::new(
        radius * cos_theta - tangential_offset * sin_theta,
        radius * sin_theta + tangential_offset * cos_theta,
        z,
    )
}

fn hub_panels(hub: &PcFanHub) -> Vec<PcFanSurfacePanel> {
    let z0 = hub.center_z_m - 0.5 * hub.depth_m;
    let z1 = hub.center_z_m + 0.5 * hub.depth_m;
    let mut panels = cylinder_panels_at(
        hub.radius_m,
        z0,
        z1,
        hub.circumferential_segments,
        PcFanSurfaceKind::Hub,
    );
    panels.extend(disk_cap_panels_at(
        hub.radius_m,
        z1,
        hub.circumferential_segments,
        PcFanSurfaceKind::Hub,
    ));
    panels
}

fn hub_panel_count(hub: &PcFanHub) -> usize {
    2 * hub.circumferential_segments
}

fn rotating_hub_z_bounds(dimensions: PcFanDimensions) -> (f64, f64) {
    let rear_z = -0.5 * dimensions.depth_m;
    let front_z = 0.5 * dimensions.depth_m;
    let support_front_z = rear_z + dimensions.support_strut_thickness_m.max(0.0);
    let hub_rear_z = (support_front_z + ROTOR_SUPPORT_CLEARANCE_M).min(front_z);

    (hub_rear_z, front_z)
}

fn shroud_panels(shroud: &PcFanShroud, notches: &[ShroudNotch]) -> Vec<PcFanSurfacePanel> {
    let z0 = -0.5 * shroud.depth_m;
    let z_top = z0 + shroud.depth_m;
    let depth = shroud.depth_m.max(f64::EPSILON);
    let axial_segments = shroud.axial_segments.max(1);

    // Build the list of axial split points so notch z bounds become
    // exact panel boundaries (otherwise notch openings get rounded to
    // the nearest segment edge).
    let mut z_boundaries: Vec<f64> = (0..=axial_segments)
        .map(|iz| z0 + depth * iz as f64 / axial_segments as f64)
        .collect();
    for notch in notches {
        for &z_split in &[notch.z_min, notch.z_max] {
            if z_split > z0 + 1.0e-9
                && z_split < z_top - 1.0e-9
                && !z_boundaries.iter().any(|&z| (z - z_split).abs() < 1.0e-9)
            {
                z_boundaries.push(z_split);
            }
        }
    }
    z_boundaries.sort_by(|a, b| a.total_cmp(b));

    let mut panels = Vec::with_capacity(shroud_panel_count(shroud));

    // Inner and outer cylinder bands.
    for window in z_boundaries.windows(2) {
        let z_start = window[0];
        let z_end = window[1];
        let z_mid = 0.5 * (z_start + z_end);
        let t0 = ((z_start - z0) / depth).clamp(0.0, 1.0);
        let t1 = ((z_end - z0) / depth).clamp(0.0, 1.0);
        let inner_r0 = shroud.radius_at_depth_fraction(t0);
        let inner_r1 = shroud.radius_at_depth_fraction(t1);
        let outer_r0 = shroud.outer_radius_at_depth_fraction(t0);
        let outer_r1 = shroud.outer_radius_at_depth_fraction(t1);

        let angles = shroud_panel_angles(shroud.circumferential_segments, z_mid, notches);
        for window in angles.windows(2) {
            let a0 = window[0];
            let a1 = window[1];
            if shroud_in_notch(0.5 * (a0 + a1), z_mid, notches) {
                continue;
            }

            panels.push(surface_panel(
                [
                    polar_point(inner_r0, a0, 0.0, z_start),
                    polar_point(inner_r0, a1, 0.0, z_start),
                    polar_point(inner_r1, a1, 0.0, z_end),
                    polar_point(inner_r1, a0, 0.0, z_end),
                ],
                PcFanSurfaceKind::Shroud,
            ));
            panels.push(surface_panel(
                [
                    polar_point(outer_r0, a1, 0.0, z_start),
                    polar_point(outer_r0, a0, 0.0, z_start),
                    polar_point(outer_r1, a0, 0.0, z_end),
                    polar_point(outer_r1, a1, 0.0, z_end),
                ],
                PcFanSurfaceKind::Shroud,
            ));
        }
    }

    // End caps: skip angular notch only at z's that lie inside a notch.
    for (depth_fraction, z) in [(0.0, z0), (1.0, z_top)] {
        let inner_r = shroud.radius_at_depth_fraction(depth_fraction);
        let outer_r = shroud.outer_radius_at_depth_fraction(depth_fraction);

        let angles = shroud_panel_angles(shroud.circumferential_segments, z, notches);
        for window in angles.windows(2) {
            let a0 = window[0];
            let a1 = window[1];
            if shroud_in_notch(0.5 * (a0 + a1), z, notches) {
                continue;
            }
            panels.push(surface_panel(
                [
                    polar_point(inner_r, a0, 0.0, z),
                    polar_point(inner_r, a1, 0.0, z),
                    polar_point(outer_r, a1, 0.0, z),
                    polar_point(outer_r, a0, 0.0, z),
                ],
                PcFanSurfaceKind::Shroud,
            ));
        }
    }

    // Slot ceiling: the inside surface of the shroud above each notch.
    // Faces -z so it caps the top of the strut slot (the slot opens
    // through the back face at z = z0 = z_min). Subdivide angularly so
    // the spline radius is sampled at notch edges and at the segment
    // grid, keeping it consistent with the cylinder panels.
    for notch in notches {
        let half = notch.half_width_rad;
        let z_ceiling = notch.z_max;
        if z_ceiling <= z0 + 1.0e-9 || z_ceiling >= z_top - 1.0e-9 {
            continue;
        }
        let t = ((z_ceiling - z0) / depth).clamp(0.0, 1.0);
        let inner_r = shroud.radius_at_depth_fraction(t);
        let outer_r = shroud.outer_radius_at_depth_fraction(t);
        let segments = ((shroud.circumferential_segments as f64 * (2.0 * half)
            / std::f64::consts::TAU)
            .ceil() as usize)
            .max(2);
        for i in 0..segments {
            let a0 = notch.angle_rad - half + 2.0 * half * i as f64 / segments as f64;
            let a1 = notch.angle_rad - half + 2.0 * half * (i + 1) as f64 / segments as f64;
            panels.push(surface_panel(
                [
                    polar_point(outer_r, a0, 0.0, z_ceiling),
                    polar_point(inner_r, a0, 0.0, z_ceiling),
                    polar_point(inner_r, a1, 0.0, z_ceiling),
                    polar_point(outer_r, a1, 0.0, z_ceiling),
                ],
                PcFanSurfaceKind::Shroud,
            ));
        }
    }

    // Slot side walls: at each notch's two angular edges, spanning
    // [notch.z_min, notch.z_max] only — not the full shroud depth.
    // The two walls face opposite tangential directions; the leading-
    // edge wall is wound in reverse so its outward normal points into
    // the empty slot volume.
    for notch in notches {
        let half = notch.half_width_rad;
        let wall_z_min = notch.z_min.max(z0);
        let wall_z_max = notch.z_max.min(z_top);
        let wall_z_pairs: Vec<(f64, f64)> = z_boundaries
            .iter()
            .copied()
            .filter(|&z| z >= wall_z_min - 1.0e-9 && z <= wall_z_max + 1.0e-9)
            .collect::<Vec<_>>()
            .windows(2)
            .map(|w| (w[0], w[1]))
            .collect();
        for (edge_index, edge_angle) in [notch.angle_rad - half, notch.angle_rad + half]
            .into_iter()
            .enumerate()
        {
            for &(z_a, z_b) in &wall_z_pairs {
                let t_a = ((z_a - z0) / depth).clamp(0.0, 1.0);
                let t_b = ((z_b - z0) / depth).clamp(0.0, 1.0);
                let inner_a = shroud.radius_at_depth_fraction(t_a);
                let outer_a = shroud.outer_radius_at_depth_fraction(t_a);
                let inner_b = shroud.radius_at_depth_fraction(t_b);
                let outer_b = shroud.outer_radius_at_depth_fraction(t_b);
                let mut vertices = [
                    polar_point(inner_a, edge_angle, 0.0, z_a),
                    polar_point(outer_a, edge_angle, 0.0, z_a),
                    polar_point(outer_b, edge_angle, 0.0, z_b),
                    polar_point(inner_b, edge_angle, 0.0, z_b),
                ];
                if edge_index == 0 {
                    vertices.reverse();
                }
                panels.push(surface_panel(vertices, PcFanSurfaceKind::Shroud));
            }
        }
    }

    panels
}

fn shroud_panel_count(shroud: &PcFanShroud) -> usize {
    (2 * shroud.axial_segments.max(1) + 2) * shroud.circumferential_segments
}

fn cylinder_panels_at(
    radius_m: f64,
    z0: f64,
    z1: f64,
    segments: usize,
    kind: PcFanSurfaceKind,
) -> Vec<PcFanSurfacePanel> {
    (0..segments)
        .map(|i| {
            let a0 = std::f64::consts::TAU * i as f64 / segments as f64;
            let a1 = std::f64::consts::TAU * (i + 1) as f64 / segments as f64;
            let vertices = [
                polar_point(radius_m, a0, 0.0, z0),
                polar_point(radius_m, a1, 0.0, z0),
                polar_point(radius_m, a1, 0.0, z1),
                polar_point(radius_m, a0, 0.0, z1),
            ];
            PcFanSurfacePanel {
                vertices,
                kind,
                blade_index: None,
                span_fraction: 0.0,
                area: quad_area(vertices),
            }
        })
        .collect()
}

fn frame_box(dimensions: PcFanDimensions, shroud: &PcFanShroud) -> PcFanFrameBox {
    let outer_side_m = dimensions.frame_side_m;
    let shroud_outer_diameter_m = 2.0 * shroud.outer_radius_m;
    let inner_side_m = (outer_side_m - 2.0 * dimensions.frame_wall_thickness_m)
        .max(shroud_outer_diameter_m)
        .min(outer_side_m);

    PcFanFrameBox {
        outer_side_m,
        inner_side_m,
        depth_m: dimensions.depth_m,
        wall_thickness_m: 0.5 * (outer_side_m - inner_side_m),
    }
}

fn frame_box_panel_count() -> usize {
    16
}

fn frame_box_panels(frame: &PcFanFrameBox) -> Vec<PcFanSurfacePanel> {
    let outer = 0.5 * frame.outer_side_m;
    let inner = 0.5 * frame.inner_side_m;
    let z0 = -0.5 * frame.depth_m;
    let z1 = 0.5 * frame.depth_m;
    let mut panels = Vec::with_capacity(frame_box_panel_count());

    for z in [z0, z1] {
        panels.push(rect_xy(
            -outer,
            outer,
            inner,
            outer,
            z,
            PcFanSurfaceKind::FrameBox,
        ));
        panels.push(rect_xy(
            -outer,
            outer,
            -outer,
            -inner,
            z,
            PcFanSurfaceKind::FrameBox,
        ));
        panels.push(rect_xy(
            -outer,
            -inner,
            -inner,
            inner,
            z,
            PcFanSurfaceKind::FrameBox,
        ));
        panels.push(rect_xy(
            inner,
            outer,
            -inner,
            inner,
            z,
            PcFanSurfaceKind::FrameBox,
        ));
    }

    panels.push(rect_xz(
        -outer,
        outer,
        z0,
        z1,
        outer,
        PcFanSurfaceKind::FrameBox,
    ));
    panels.push(rect_xz(
        -outer,
        outer,
        z0,
        z1,
        -outer,
        PcFanSurfaceKind::FrameBox,
    ));
    panels.push(rect_yz(
        -outer,
        outer,
        z0,
        z1,
        outer,
        PcFanSurfaceKind::FrameBox,
    ));
    panels.push(rect_yz(
        -outer,
        outer,
        z0,
        z1,
        -outer,
        PcFanSurfaceKind::FrameBox,
    ));

    panels.push(rect_xz(
        -inner,
        inner,
        z0,
        z1,
        inner,
        PcFanSurfaceKind::FrameBox,
    ));
    panels.push(rect_xz(
        -inner,
        inner,
        z0,
        z1,
        -inner,
        PcFanSurfaceKind::FrameBox,
    ));
    panels.push(rect_yz(
        -inner,
        inner,
        z0,
        z1,
        inner,
        PcFanSurfaceKind::FrameBox,
    ));
    panels.push(rect_yz(
        -inner,
        inner,
        z0,
        z1,
        -inner,
        PcFanSurfaceKind::FrameBox,
    ));

    panels
}

fn rect_xy(
    x0: f64,
    x1: f64,
    y0: f64,
    y1: f64,
    z: f64,
    kind: PcFanSurfaceKind,
) -> PcFanSurfacePanel {
    let vertices = [
        Point3::new(x0, y0, z),
        Point3::new(x1, y0, z),
        Point3::new(x1, y1, z),
        Point3::new(x0, y1, z),
    ];
    surface_panel(vertices, kind)
}

fn rect_xz(
    x0: f64,
    x1: f64,
    z0: f64,
    z1: f64,
    y: f64,
    kind: PcFanSurfaceKind,
) -> PcFanSurfacePanel {
    let vertices = [
        Point3::new(x0, y, z0),
        Point3::new(x1, y, z0),
        Point3::new(x1, y, z1),
        Point3::new(x0, y, z1),
    ];
    surface_panel(vertices, kind)
}

fn rect_yz(
    y0: f64,
    y1: f64,
    z0: f64,
    z1: f64,
    x: f64,
    kind: PcFanSurfaceKind,
) -> PcFanSurfacePanel {
    let vertices = [
        Point3::new(x, y0, z0),
        Point3::new(x, y1, z0),
        Point3::new(x, y1, z1),
        Point3::new(x, y0, z1),
    ];
    surface_panel(vertices, kind)
}

fn fixed_frame_mounts(dimensions: PcFanDimensions) -> [FrameMountPoint; FRAME_MOUNT_COUNT] {
    let half_span = 0.5 * dimensions.frame_side_m - dimensions.frame_mount_inset_m;
    [
        FrameMountPoint {
            position: Point3::new(-half_span, -half_span, 0.0),
            pad_size_m: dimensions.mount_pad_size_m,
        },
        FrameMountPoint {
            position: Point3::new(half_span, -half_span, 0.0),
            pad_size_m: dimensions.mount_pad_size_m,
        },
        FrameMountPoint {
            position: Point3::new(half_span, half_span, 0.0),
            pad_size_m: dimensions.mount_pad_size_m,
        },
        FrameMountPoint {
            position: Point3::new(-half_span, half_span, 0.0),
            pad_size_m: dimensions.mount_pad_size_m,
        },
    ]
}

fn support_struts(
    dimensions: PcFanDimensions,
    frame: &PcFanFrameBox,
) -> [PcFanSupportStrut; SUPPORT_STRUT_COUNT] {
    let z = -0.5 * dimensions.depth_m + dimensions.support_strut_thickness_m.max(0.0);
    let half_width = 0.5 * dimensions.support_strut_width_m;
    let inner_half = (0.5 * frame.inner_side_m)
        .min(0.5 * frame.outer_side_m - half_width * std::f64::consts::FRAC_1_SQRT_2)
        .max(0.0);
    let center = Point3::new(0.0, 0.0, z);
    [
        PcFanSupportStrut {
            start: Point3::new(-inner_half, -inner_half, z),
            end: center,
            width_m: dimensions.support_strut_width_m,
            thickness_m: dimensions.support_strut_thickness_m,
        },
        PcFanSupportStrut {
            start: Point3::new(inner_half, inner_half, z),
            end: center,
            width_m: dimensions.support_strut_width_m,
            thickness_m: dimensions.support_strut_thickness_m,
        },
        PcFanSupportStrut {
            start: Point3::new(-inner_half, inner_half, z),
            end: center,
            width_m: dimensions.support_strut_width_m,
            thickness_m: dimensions.support_strut_thickness_m,
        },
        PcFanSupportStrut {
            start: Point3::new(inner_half, -inner_half, z),
            end: center,
            width_m: dimensions.support_strut_width_m,
            thickness_m: dimensions.support_strut_thickness_m,
        },
    ]
}

fn support_strut_panel_count() -> usize {
    10
}

fn support_strut_panels(strut: &PcFanSupportStrut) -> Vec<PcFanSurfacePanel> {
    let dx = strut.end.x - strut.start.x;
    let dy = strut.end.y - strut.start.y;
    let length = dx.hypot(dy).max(f64::EPSILON);
    let half_width = 0.5 * strut.width_m;
    let bevel = half_width.min(0.5 * length);

    let ux = dx / length;
    let uy = dy / length;
    let nx = -uy;
    let ny = ux;

    let max_x = strut.start.x + bevel * ux;
    let max_y = strut.start.y + bevel * uy;
    let mbx = strut.end.x - bevel * ux;
    let mby = strut.end.y - bevel * uy;

    let z1 = strut.start.z;
    let z0 = z1 - strut.thickness_m;

    let v0_t = Point3::new(strut.start.x, strut.start.y, z1);
    let v1_t = Point3::new(max_x + half_width * nx, max_y + half_width * ny, z1);
    let v2_t = Point3::new(mbx + half_width * nx, mby + half_width * ny, z1);
    let v3_t = Point3::new(strut.end.x, strut.end.y, z1);
    let v4_t = Point3::new(mbx - half_width * nx, mby - half_width * ny, z1);
    let v5_t = Point3::new(max_x - half_width * nx, max_y - half_width * ny, z1);

    let v0_b = Point3::new(v0_t.x, v0_t.y, z0);
    let v1_b = Point3::new(v1_t.x, v1_t.y, z0);
    let v2_b = Point3::new(v2_t.x, v2_t.y, z0);
    let v3_b = Point3::new(v3_t.x, v3_t.y, z0);
    let v4_b = Point3::new(v4_t.x, v4_t.y, z0);
    let v5_b = Point3::new(v5_t.x, v5_t.y, z0);

    vec![
        surface_panel([v0_t, v5_t, v4_t, v3_t], PcFanSurfaceKind::SupportStrut),
        surface_panel([v0_t, v3_t, v2_t, v1_t], PcFanSurfaceKind::SupportStrut),
        surface_panel([v0_b, v1_b, v2_b, v3_b], PcFanSurfaceKind::SupportStrut),
        surface_panel([v0_b, v3_b, v4_b, v5_b], PcFanSurfaceKind::SupportStrut),
        surface_panel([v0_t, v0_b, v5_b, v5_t], PcFanSurfaceKind::SupportStrut),
        surface_panel([v5_t, v5_b, v4_b, v4_t], PcFanSurfaceKind::SupportStrut),
        surface_panel([v4_t, v4_b, v3_b, v3_t], PcFanSurfaceKind::SupportStrut),
        surface_panel([v3_t, v3_b, v2_b, v2_t], PcFanSurfaceKind::SupportStrut),
        surface_panel([v2_t, v2_b, v1_b, v1_t], PcFanSurfaceKind::SupportStrut),
        surface_panel([v1_t, v1_b, v0_b, v0_t], PcFanSurfaceKind::SupportStrut),
    ]
}

fn motor_mount(dimensions: PcFanDimensions, circumferential_segments: usize) -> PcFanMotorMount {
    let z = -0.5 * dimensions.depth_m + dimensions.support_strut_thickness_m.max(0.0);
    let max_depth_m = (0.5 * dimensions.depth_m - z).max(0.0);

    PcFanMotorMount {
        center: Point3::new(0.0, 0.0, z),
        radius_m: dimensions.motor_mount_radius_m,
        depth_m: dimensions.motor_mount_depth_m.min(max_depth_m),
        circumferential_segments,
    }
}

fn disk_cap_panels_at(
    radius_m: f64,
    z: f64,
    segments: usize,
    kind: PcFanSurfaceKind,
) -> Vec<PcFanSurfacePanel> {
    let center = Point3::new(0.0, 0.0, z);
    (0..segments)
        .map(|i| {
            let a0 = std::f64::consts::TAU * i as f64 / segments as f64;
            let a1 = std::f64::consts::TAU * (i + 1) as f64 / segments as f64;
            surface_panel(
                [
                    center,
                    polar_point(radius_m, a0, 0.0, z),
                    polar_point(radius_m, a1, 0.0, z),
                    center,
                ],
                kind,
            )
        })
        .collect()
}

fn surface_panel(vertices: [Point3; 4], kind: PcFanSurfaceKind) -> PcFanSurfacePanel {
    PcFanSurfacePanel {
        vertices,
        kind,
        blade_index: None,
        span_fraction: 0.0,
        area: quad_area(vertices),
    }
}

#[derive(Debug, Clone, Copy)]
struct BladeCollisionReport {
    invalid_intersection_count: usize,
    minimum_blade_gap_m: f64,
}

#[derive(Debug, Clone, Copy)]
struct BladePanelBounds {
    blade_index: usize,
    min_radius_m: f64,
    max_radius_m: f64,
    mean_radius_m: f64,
    min_z_m: f64,
    max_z_m: f64,
    center_angle_rad: f64,
    angular_half_width_rad: f64,
}

impl BladePanelBounds {
    fn from_panel(panel: &PcFanBladePanel) -> Self {
        let mut min_radius_m = f64::INFINITY;
        let mut max_radius_m: f64 = 0.0;
        let mut radius_sum_m = 0.0;
        let mut min_z_m = f64::INFINITY;
        let mut max_z_m = f64::NEG_INFINITY;
        let mut sin_sum = 0.0;
        let mut cos_sum = 0.0;
        let mut angles = [0.0; 4];

        for (i, vertex) in panel.vertices.iter().enumerate() {
            let radius_m = (vertex.x * vertex.x + vertex.y * vertex.y).sqrt();
            let angle = vertex.y.atan2(vertex.x);
            angles[i] = angle;
            min_radius_m = min_radius_m.min(radius_m);
            max_radius_m = max_radius_m.max(radius_m);
            radius_sum_m += radius_m;
            min_z_m = min_z_m.min(vertex.z);
            max_z_m = max_z_m.max(vertex.z);
            sin_sum += angle.sin();
            cos_sum += angle.cos();
        }

        let center_angle_rad = if sin_sum.hypot(cos_sum) < f64::EPSILON {
            angles[0]
        } else {
            sin_sum.atan2(cos_sum)
        };
        let angular_half_width_rad = angles
            .iter()
            .map(|&angle| angular_distance(angle, center_angle_rad))
            .fold(0.0, f64::max);
        let half_thickness_m = 0.5 * panel.thickness_m.max(0.0);

        Self {
            blade_index: panel.blade_index,
            min_radius_m,
            max_radius_m,
            mean_radius_m: 0.25 * radius_sum_m,
            min_z_m: min_z_m - half_thickness_m,
            max_z_m: max_z_m + half_thickness_m,
            center_angle_rad,
            angular_half_width_rad,
        }
    }
}

fn blade_collision_report(geometry: &PcFanGeometry) -> BladeCollisionReport {
    let bounds = geometry
        .blades
        .iter()
        .flat_map(|blade| blade.panels.iter().map(BladePanelBounds::from_panel))
        .collect::<Vec<_>>();
    let root_merge_radius_m = root_merge_radius_m(geometry, &bounds);
    let mut invalid_intersection_count = 0;
    let mut minimum_blade_gap_m = f64::INFINITY;

    for i in 0..bounds.len() {
        for j in (i + 1)..bounds.len() {
            let a = bounds[i];
            let b = bounds[j];
            if a.blade_index == b.blade_index {
                continue;
            }

            let gap_m = blade_panel_gap_m(a, b);
            minimum_blade_gap_m = minimum_blade_gap_m.min(gap_m);
            if gap_m <= COLLISION_EPSILON_M
                && !(a.max_radius_m <= root_merge_radius_m && b.max_radius_m <= root_merge_radius_m)
            {
                invalid_intersection_count += 1;
            }
        }
    }

    BladeCollisionReport {
        invalid_intersection_count,
        minimum_blade_gap_m: if minimum_blade_gap_m.is_finite() {
            minimum_blade_gap_m
        } else {
            0.0
        },
    }
}

fn root_merge_radius_m(geometry: &PcFanGeometry, bounds: &[BladePanelBounds]) -> f64 {
    let max_blade_radius_m = bounds
        .iter()
        .map(|bounds| bounds.max_radius_m)
        .fold(geometry.hub.radius_m, f64::max);
    geometry.hub.radius_m
        + ROOT_MERGE_SPAN_FRACTION * (max_blade_radius_m - geometry.hub.radius_m).max(0.0)
}

fn blade_panel_gap_m(a: BladePanelBounds, b: BladePanelBounds) -> f64 {
    let radial_gap_m = interval_gap(
        a.min_radius_m,
        a.max_radius_m,
        b.min_radius_m,
        b.max_radius_m,
    );
    let axial_gap_m = interval_gap(a.min_z_m, a.max_z_m, b.min_z_m, b.max_z_m);
    let angular_gap_rad = (angular_distance(a.center_angle_rad, b.center_angle_rad)
        - a.angular_half_width_rad
        - b.angular_half_width_rad)
        .max(0.0);
    let angular_gap_m = angular_gap_rad * a.mean_radius_m.min(b.mean_radius_m);

    (radial_gap_m * radial_gap_m + axial_gap_m * axial_gap_m + angular_gap_m * angular_gap_m).sqrt()
}

fn interval_gap(a0: f64, a1: f64, b0: f64, b1: f64) -> f64 {
    if a1 < b0 {
        b0 - a1
    } else if b1 < a0 {
        a0 - b1
    } else {
        0.0
    }
}

fn quad_area(vertices: [Point3; 4]) -> f64 {
    triangle_area(vertices[0], vertices[1], vertices[2])
        + triangle_area(vertices[0], vertices[2], vertices[3])
}

fn triangle_area(a: Point3, b: Point3, c: Point3) -> f64 {
    0.5 * b.sub(a).cross(c.sub(a)).norm()
}

fn angular_distance(a: f64, b: f64) -> f64 {
    let d = (a - b).abs().rem_euclid(std::f64::consts::TAU);
    d.min(std::f64::consts::TAU - d)
}

fn station_fraction(index: usize, count: usize) -> f64 {
    if count <= 1 {
        0.0
    } else {
        index as f64 / (count - 1) as f64
    }
}

fn sample_spline<T>(controls: &[T], t: f64) -> T
where
    T: Copy + SplineInterpolate,
{
    let t = t.clamp(0.0, 1.0);
    let scaled = t * (controls.len() - 1) as f64;
    let i1 = scaled.floor() as usize;
    let i2 = (i1 + 1).min(controls.len() - 1);
    let i0 = i1.saturating_sub(1);
    let i3 = (i2 + 1).min(controls.len() - 1);
    let local_t = scaled - i1 as f64;
    T::catmull_rom(
        controls[i0],
        controls[i1],
        controls[i2],
        controls[i3],
        local_t,
    )
}

fn sample_spline_derivative(controls: &[f64], t: f64) -> f64 {
    if controls.len() <= 1 {
        return 0.0;
    }
    let t = t.clamp(0.0, 1.0);
    let scaled = t * (controls.len() - 1) as f64;
    let i1 = scaled.floor() as usize;
    let i2 = (i1 + 1).min(controls.len() - 1);
    let i0 = i1.saturating_sub(1);
    let i3 = (i2 + 1).min(controls.len() - 1);
    let local_t = scaled - i1 as f64;
    let derivative = 0.5
        * ((-controls[i0] + controls[i2])
            + 2.0
                * (2.0 * controls[i0] - 5.0 * controls[i1] + 4.0 * controls[i2] - controls[i3])
                * local_t
            + 3.0
                * (-controls[i0] + 3.0 * controls[i1] - 3.0 * controls[i2] + controls[i3])
                * local_t
                * local_t);
    derivative * (controls.len() - 1) as f64
}

trait SplineInterpolate {
    fn catmull_rom(p0: Self, p1: Self, p2: Self, p3: Self, t: f64) -> Self;
}

impl SplineInterpolate for BladePathControl {
    fn catmull_rom(p0: Self, p1: Self, p2: Self, p3: Self, t: f64) -> Self {
        Self {
            tangential_offset_m: catmull_rom_scalar(
                p0.tangential_offset_m,
                p1.tangential_offset_m,
                p2.tangential_offset_m,
                p3.tangential_offset_m,
                t,
            ),
            axial_offset_m: catmull_rom_scalar(
                p0.axial_offset_m,
                p1.axial_offset_m,
                p2.axial_offset_m,
                p3.axial_offset_m,
                t,
            ),
        }
    }
}

impl SplineInterpolate for BladeSectionControl {
    fn catmull_rom(p0: Self, p1: Self, p2: Self, p3: Self, t: f64) -> Self {
        Self {
            chord_m: catmull_rom_scalar(p0.chord_m, p1.chord_m, p2.chord_m, p3.chord_m, t),
            camber_m: catmull_rom_scalar(p0.camber_m, p1.camber_m, p2.camber_m, p3.camber_m, t),
            thickness_m: catmull_rom_scalar(
                p0.thickness_m,
                p1.thickness_m,
                p2.thickness_m,
                p3.thickness_m,
                t,
            ),
            twist_deg: catmull_rom_scalar(
                p0.twist_deg,
                p1.twist_deg,
                p2.twist_deg,
                p3.twist_deg,
                t,
            ),
        }
    }
}

impl SplineInterpolate for f64 {
    fn catmull_rom(p0: Self, p1: Self, p2: Self, p3: Self, t: f64) -> Self {
        catmull_rom_scalar(p0, p1, p2, p3, t)
    }
}

fn catmull_rom_scalar(p0: f64, p1: f64, p2: f64, p3: f64, t: f64) -> f64 {
    let t2 = t * t;
    let t3 = t2 * t;
    0.5 * ((2.0 * p1)
        + (-p0 + p2) * t
        + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2
        + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3)
}

struct PcFanVtuView<'a> {
    panels: &'a [PcFanSurfacePanel],
}

impl MeshView for PcFanVtuView<'_> {
    fn num_nodes(&self) -> usize {
        self.panels.len() * 4
    }

    fn num_cells(&self) -> usize {
        self.panels.len()
    }

    fn nodes(&self) -> Box<dyn Iterator<Item = [f64; 3]> + '_> {
        Box::new(
            self.panels
                .iter()
                .flat_map(|panel| panel.vertices.iter().map(|p| [p.x, p.y, p.z])),
        )
    }

    fn cells(&self) -> Box<dyn Iterator<Item = (CellKind, Vec<u64>)> + '_> {
        Box::new((0..self.panels.len()).map(|i| {
            let base = (i * 4) as u64;
            (CellKind::Quad, vec![base, base + 1, base + 2, base + 3])
        }))
    }
}

struct PcFanCfdDomainVtuView<'a> {
    domain: &'a PcFanCfdDomain,
}

impl MeshView for PcFanCfdDomainVtuView<'_> {
    fn num_nodes(&self) -> usize {
        self.domain.nodes.len()
    }

    fn num_cells(&self) -> usize {
        self.domain.cells.len()
    }

    fn nodes(&self) -> Box<dyn Iterator<Item = [f64; 3]> + '_> {
        Box::new(self.domain.nodes.iter().map(|p| [p.x, p.y, p.z]))
    }

    fn cells(&self) -> Box<dyn Iterator<Item = (CellKind, Vec<u64>)> + '_> {
        Box::new(self.domain.cells.iter().map(|cell| {
            (
                CellKind::Hexahedron,
                cell.node_indices
                    .iter()
                    .map(|index| *index as u64)
                    .collect(),
            )
        }))
    }
}

struct PcFanCfdBoundaryVtuView<'a> {
    domain: &'a PcFanCfdDomain,
}

impl MeshView for PcFanCfdBoundaryVtuView<'_> {
    fn num_nodes(&self) -> usize {
        self.domain.boundary_faces.len() * 4
    }

    fn num_cells(&self) -> usize {
        self.domain.boundary_faces.len()
    }

    fn nodes(&self) -> Box<dyn Iterator<Item = [f64; 3]> + '_> {
        Box::new(
            self.domain
                .boundary_faces
                .iter()
                .flat_map(|face| face.vertices.iter().map(|p| [p.x, p.y, p.z])),
        )
    }

    fn cells(&self) -> Box<dyn Iterator<Item = (CellKind, Vec<u64>)> + '_> {
        Box::new((0..self.domain.boundary_faces.len()).map(|i| {
            let base = (i * 4) as u64;
            (CellKind::Quad, vec![base, base + 1, base + 2, base + 3])
        }))
    }
}

struct PcFanCfdFlowSolutionVtuView<'a> {
    solution: &'a PcFanCfdFlowSolution,
}

impl MeshView for PcFanCfdFlowSolutionVtuView<'_> {
    fn num_nodes(&self) -> usize {
        self.solution.nodes.len()
    }

    fn num_cells(&self) -> usize {
        self.solution.domain.cells.len()
    }

    fn nodes(&self) -> Box<dyn Iterator<Item = [f64; 3]> + '_> {
        Box::new(self.solution.nodes.iter().map(|p| [p.x, p.y, p.z]))
    }

    fn cells(&self) -> Box<dyn Iterator<Item = (CellKind, Vec<u64>)> + '_> {
        Box::new(self.solution.domain.cells.iter().map(|cell| {
            (
                CellKind::Hexahedron,
                cell.node_indices
                    .iter()
                    .map(|index| *index as u64)
                    .collect(),
            )
        }))
    }
}

#[cfg(test)]
#[allow(clippy::items_after_test_module)] // CFD-export helpers below are shared non-test code.
mod input_validation_tests {
    use super::*;

    #[test]
    fn geometry_rejects_non_finite_dimensions_before_building() {
        let mut config = PcFanConfig::default();
        let design = config.default_design_variables();
        config.dimensions.support_strut_width_m = f64::NAN;

        assert_eq!(
            config.try_build_geometry(&design),
            Err(PcFanGeometryInputError::NonFiniteDimension(
                "support_strut_width_m"
            ))
        );
    }

    #[test]
    fn geometry_rejects_non_finite_design_variables_before_building() {
        let config = PcFanConfig::default();
        let mut design = config.default_design_variables();
        design.shroud_radius_offsets_m[1] = f64::INFINITY;
        let expected_index = DESIGN_VARIABLE_COUNT - SHROUD_CONTROL_COUNT + 1;

        assert_eq!(
            config.try_build_geometry(&design),
            Err(PcFanGeometryInputError::NonFiniteDesignVariable(
                expected_index
            ))
        );
    }

    #[test]
    fn geometry_rejects_nonphysical_fixed_dimensions_before_building() {
        let mut config = PcFanConfig::default();
        let design = config.default_design_variables();
        config.dimensions.blade_tip_radius_m = config.dimensions.hub_radius_m;

        assert!(matches!(
            config.try_build_geometry(&design),
            Err(PcFanGeometryInputError::InvalidDimensionRelation {
                requirement: "hub_radius_m < blade_tip_radius_m < shroud_inner_radius_m"
            })
        ));

        let mut config = PcFanConfig::default();
        let design = config.default_design_variables();
        config.dimensions.strut_wake_centerline_deficit = 1.1;
        assert!(matches!(
            config.try_build_geometry(&design),
            Err(PcFanGeometryInputError::InvalidDimension {
                field: "strut_wake_centerline_deficit",
                ..
            })
        ));
    }

    #[test]
    fn radial_envelope_design_controls_round_trip_and_move_geometry() {
        let config = PcFanConfig::default_120mm();
        let baseline = config.default_design_variables();
        let mut values = baseline.to_vector();
        let radial_start = BLADE_PATH_CONTROL_COUNT * 2 + BLADE_SECTION_STATION_COUNT * 4;
        values[radial_start] = 0.0015;
        values[radial_start + 1] = -0.0010;

        let design = PcFanDesignVariables::from_vector(values);
        assert_eq!(design.to_vector(), values);
        assert_eq!(design.hub_radius_offset_m, 0.0015);
        assert_eq!(design.tip_radius_offset_m, -0.0010);

        let geometry = config
            .try_build_geometry(&design)
            .expect("bounded radial-envelope design should be admissible");
        assert!((geometry.hub.radius_m - 0.0195).abs() < 1.0e-12);
        assert!((geometry.blade_tip_radius_m - 0.0530).abs() < 1.0e-12);
        assert!(geometry.hub.radius_m < geometry.blade_tip_radius_m);
        assert!(geometry.blade_tip_radius_m < geometry.shroud.minimum_radius_m());
    }

    #[test]
    fn radial_envelope_design_rejects_collapsed_effective_gap() {
        let config = PcFanConfig::default_120mm();
        let mut design = config.default_design_variables();
        design.tip_radius_offset_m = -0.036;

        assert!(matches!(
            config.try_build_geometry(&design),
            Err(PcFanGeometryInputError::InvalidDimensionRelation { .. })
        ));
    }

    #[test]
    fn default_fixed_geometry_configuration_is_admissible() {
        let config = PcFanConfig::default();
        assert!(config.validate_geometry_config().is_ok());
        assert!(
            config
                .try_build_geometry(&config.default_design_variables())
                .is_ok()
        );
    }

    #[test]
    fn fixed_geometry_validation_rejects_non_finite_values_directly() {
        let mut config = PcFanConfig::default();
        config.dimensions.frame_side_m = f64::NAN;

        assert!(matches!(
            config.validate_geometry_config(),
            Err(PcFanGeometryInputError::InvalidDimension {
                field: "frame_side_m",
                ..
            })
        ));
    }
}

fn nearest_solid_boundary_fields(
    nodes: &[Point3],
    boundary_faces: &[PcFanCfdBoundaryFace],
) -> (Vec<f64>, Vec<f64>) {
    let solid_centers = boundary_faces
        .iter()
        .filter(|face| face.source_surface.is_some() && face.area > 0.0)
        .map(|face| (boundary_face_center(face), face.kind.boundary_id()))
        .collect::<Vec<_>>();

    nodes
        .iter()
        .map(|node| {
            solid_centers
                .iter()
                .map(|(center, kind)| (*kind, point_distance_m(*node, *center)))
                .min_by(|(_, a), (_, b)| a.total_cmp(b))
                .unwrap_or((-1.0, f64::INFINITY))
        })
        .unzip()
}

fn boundary_face_center(face: &PcFanCfdBoundaryFace) -> Point3 {
    Point3::new(
        0.25 * face.vertices.iter().map(|point| point.x).sum::<f64>(),
        0.25 * face.vertices.iter().map(|point| point.y).sum::<f64>(),
        0.25 * face.vertices.iter().map(|point| point.z).sum::<f64>(),
    )
}

fn point_distance_m(a: Point3, b: Point3) -> f64 {
    let dx = a.x - b.x;
    let dy = a.y - b.y;
    let dz = a.z - b.z;

    (dx * dx + dy * dy + dz * dz).sqrt()
}
