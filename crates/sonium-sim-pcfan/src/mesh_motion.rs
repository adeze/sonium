//! Checked topology-preserving ALE motion for the resolved PC-fan mesh.
//!
//! The production CFD discretisation is an immersed-solid tetrahedral box
//! mesh, so changing hub, tip, or shroud radii does not require changing
//! connectivity.  This module nevertheless moves the FEM coordinates in a
//! smooth, compactly supported radial field so shape perturbations are not
//! silently treated as a fixed-coordinate solve.  The adapter commits the
//! update transactionally and preserves the legacy orientation convention of
//! the FEM box-mesh generator.
//!
//! This is an ALE mesh-motion contribution, not a body-fitted remesher and
//! not an exact shape-adjoint implementation.  The checked finite-difference
//! objective path can use it directly; exact adjoint callers must still
//! publish a matching mesh-motion residual derivative.

use crate::model::{
    BLADE_PATH_CONTROL_COUNT, BLADE_SECTION_STATION_COUNT, DESIGN_VARIABLE_COUNT, PcFanConfig,
    PcFanDesignVariables, PcFanGeometry, RADIAL_ENVELOPE_CONTROL_COUNT, SHROUD_CONTROL_COUNT,
};
use sonium_fem::mesh::Mesh;
use std::fmt;

const MOTION_EPSILON_M: f64 = 1.0e-14;
const MIN_SUPPORT_WIDTH_M: f64 = 1.0e-3;

/// Error raised before a proposed PC-fan ALE update is committed.
#[derive(Debug, Clone, PartialEq)]
pub enum PcFanMeshMotionError {
    /// A design, geometry, mesh scale, or derived support width was invalid.
    InvalidInput(&'static str),
    /// The source mesh failed connectivity admission.
    InvalidMesh,
    /// One or more elements became inverted or degenerate.
    InvalidElements(Vec<usize>),
    /// An element lost too much measure during the motion.
    ElementMeasureCollapse { element: usize, ratio: f64 },
}

impl fmt::Display for PcFanMeshMotionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidInput(field) => {
                write!(formatter, "invalid PC-fan mesh-motion input `{field}`")
            }
            Self::InvalidMesh => write!(formatter, "PC-fan mesh-motion source mesh is invalid"),
            Self::InvalidElements(elements) => {
                write!(
                    formatter,
                    "PC-fan mesh motion invalid elements {elements:?}"
                )
            }
            Self::ElementMeasureCollapse { element, ratio } => write!(
                formatter,
                "PC-fan mesh-motion element {element} measure ratio {ratio} is below the quality barrier"
            ),
        }
    }
}

impl std::error::Error for PcFanMeshMotionError {}

/// Admission report for a PC-fan mesh-motion update.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PcFanMeshMotionReport {
    /// Whether a non-zero displacement was applied.
    pub applied: bool,
    /// Number of moved FEM nodes.
    pub moved_nodes: usize,
    /// Largest requested nodal displacement in metres.
    pub max_displacement_m: f64,
    /// Minimum absolute element-measure ratio after the update.
    pub min_element_measure_ratio: f64,
    /// Maximum absolute element-measure ratio after the update.
    pub max_element_measure_ratio: f64,
}

/// Apply checked radial ALE motion induced by hub, tip, and shroud offsets.
///
/// The displacement is zero on the outer Cartesian boundary and smoothly
/// tapered outside the fan axial box.  It is therefore compatible with the
/// existing inlet/outlet and side-wall boundary identification.  Connectivity,
/// element markers, and boundary markers are preserved by the underlying
/// transactional update.
pub fn apply_pc_fan_mesh_motion(
    mesh: &mut Mesh,
    config: &PcFanConfig,
    geometry: &PcFanGeometry,
    design: &PcFanDesignVariables,
) -> Result<PcFanMeshMotionReport, PcFanMeshMotionError> {
    let offsets = [
        design.hub_radius_offset_m,
        design.tip_radius_offset_m,
        design
            .shroud_radius_offsets_m
            .iter()
            .copied()
            .fold(0.0_f64, f64::max),
        design
            .shroud_radius_offsets_m
            .iter()
            .copied()
            .fold(0.0_f64, f64::min),
    ];
    if offsets.iter().any(|value| !value.is_finite()) {
        return Err(PcFanMeshMotionError::InvalidInput("design offsets"));
    }
    if mesh.dimension != 3 || mesh.nodes.is_empty() {
        return Err(PcFanMeshMotionError::InvalidInput("three-dimensional mesh"));
    }
    if mesh
        .nodes
        .iter()
        .any(|point| !point.x.is_finite() || !point.y.is_finite() || !point.z.is_finite())
    {
        return Err(PcFanMeshMotionError::InvalidInput("mesh coordinates"));
    }
    let half_side = 0.5 * geometry.frame_box.outer_side_m + config.cfd.side_clearance_m.max(0.0);
    let fan_z0 = -0.5 * geometry.frame_box.depth_m;
    let fan_z1 = 0.5 * geometry.frame_box.depth_m;
    if !half_side.is_finite()
        || half_side <= 0.0
        || !fan_z0.is_finite()
        || !fan_z1.is_finite()
        || fan_z1 <= fan_z0
    {
        return Err(PcFanMeshMotionError::InvalidInput("mesh bounds"));
    }

    let max_offset = design
        .hub_radius_offset_m
        .abs()
        .max(design.tip_radius_offset_m.abs())
        .max(
            design
                .shroud_radius_offsets_m
                .iter()
                .map(|value| value.abs())
                .fold(0.0, f64::max),
        );
    if max_offset <= MOTION_EPSILON_M {
        return Ok(PcFanMeshMotionReport {
            applied: false,
            moved_nodes: 0,
            max_displacement_m: 0.0,
            min_element_measure_ratio: 1.0,
            max_element_measure_ratio: 1.0,
        });
    }

    let cells_x = config.cfd.cells_x.max(1) as f64;
    let cells_z = config.cfd.cells_z.max(1) as f64;
    let radial_support = (2.0 * half_side / cells_x).max(MIN_SUPPORT_WIDTH_M);
    let axial_support = ((fan_z1 - fan_z0) / cells_z).max(MIN_SUPPORT_WIDTH_M);
    let hub_radius = design.effective_hub_radius_m(config);
    let tip_radius = design.effective_tip_radius_m(config);
    if !hub_radius.is_finite()
        || !tip_radius.is_finite()
        || hub_radius <= 0.0
        || tip_radius <= hub_radius
    {
        return Err(PcFanMeshMotionError::InvalidInput(
            "effective radial envelope",
        ));
    }

    let mut displacements = Vec::with_capacity(mesh.nodes.len());
    let mut max_displacement_m = 0.0_f64;
    for point in &mesh.nodes {
        let radius = point.x.hypot(point.y);
        let radial_taper = smoothstep01((half_side - radius) / half_side);
        let axial_taper = smoothstep01((point.z - (fan_z0 - axial_support)) / axial_support)
            * smoothstep01(((fan_z1 + axial_support) - point.z) / axial_support);
        let hub_weight = smoothstep01((hub_radius + radial_support - radius) / radial_support);
        let tip_weight = bell_weight((radius - tip_radius) / radial_support);
        let depth_fraction = ((point.z - fan_z0) / (fan_z1 - fan_z0)).clamp(0.0, 1.0);
        let shroud_radius = geometry.shroud.radius_at_depth_fraction(depth_fraction);
        let shroud_weight = bell_weight((radius - shroud_radius) / radial_support);
        let shroud_offset = design.shroud_radius_offset_at(depth_fraction);
        let radial_displacement = axial_taper
            * radial_taper
            * (design.hub_radius_offset_m * hub_weight
                + design.tip_radius_offset_m * tip_weight
                + shroud_offset * shroud_weight);
        let displacement = if radius > MOTION_EPSILON_M {
            [
                radial_displacement * point.x / radius,
                radial_displacement * point.y / radius,
                0.0,
            ]
        } else {
            [0.0, 0.0, 0.0]
        };
        max_displacement_m = max_displacement_m.max(norm(displacement));
        displacements.push(displacement);
    }

    mesh.validate_connectivity()
        .map_err(|_| PcFanMeshMotionError::InvalidMesh)?;
    let original_signed_measures = (0..mesh.num_elements())
        .map(|element| signed_tetra_measure(mesh, element))
        .collect::<Vec<_>>();
    if original_signed_measures
        .iter()
        .any(|measure| measure.is_none_or(|value| !value.is_finite() || value.abs() <= 0.0))
    {
        return Err(PcFanMeshMotionError::InvalidElements(
            original_signed_measures
                .iter()
                .enumerate()
                .filter_map(|(element, measure)| match measure {
                    Some(value) if value.is_finite() && value.abs() > 0.0 => None,
                    _ => Some(element),
                })
                .collect(),
        ));
    }
    let mut candidate = mesh.clone();
    for (point, displacement) in candidate.nodes.iter_mut().zip(displacements.iter()) {
        point.x += displacement[0];
        point.y += displacement[1];
        point.z += displacement[2];
    }
    let mut min_element_measure_ratio = f64::INFINITY;
    let mut max_element_measure_ratio = 0.0_f64;
    let mut invalid_elements = Vec::new();
    for (element, original_signed_measure) in original_signed_measures.iter().copied().enumerate() {
        let Some(original_signed_measure) = original_signed_measure else {
            invalid_elements.push(element);
            continue;
        };
        let Some(moved_signed_measure) = signed_tetra_measure(&candidate, element) else {
            invalid_elements.push(element);
            continue;
        };
        if !moved_signed_measure.is_finite()
            || moved_signed_measure.abs() <= 0.0
            || moved_signed_measure.signum() != original_signed_measure.signum()
        {
            invalid_elements.push(element);
            continue;
        }
        let ratio = moved_signed_measure.abs() / original_signed_measure.abs();
        if !ratio.is_finite() || ratio <= 0.0 {
            invalid_elements.push(element);
            continue;
        }
        min_element_measure_ratio = min_element_measure_ratio.min(ratio);
        max_element_measure_ratio = max_element_measure_ratio.max(ratio);
    }
    if !invalid_elements.is_empty() {
        return Err(PcFanMeshMotionError::InvalidElements(invalid_elements));
    }
    const MIN_ELEMENT_MEASURE_RATIO: f64 = 0.20;
    if min_element_measure_ratio < MIN_ELEMENT_MEASURE_RATIO {
        let element = original_signed_measures
            .iter()
            .enumerate()
            .find_map(|(element, original_signed_measure)| {
                let Some(original_signed_measure) = original_signed_measure else {
                    return Some(element);
                };
                let Some(moved_signed_measure) = signed_tetra_measure(&candidate, element) else {
                    return Some(element);
                };
                let ratio = moved_signed_measure.abs() / original_signed_measure.abs();
                (ratio < MIN_ELEMENT_MEASURE_RATIO).then_some(element)
            })
            .unwrap_or(0);
        return Err(PcFanMeshMotionError::ElementMeasureCollapse {
            element,
            ratio: min_element_measure_ratio,
        });
    }
    candidate.invalidate_caches();
    *mesh = candidate;
    Ok(PcFanMeshMotionReport {
        applied: max_displacement_m > 0.0,
        moved_nodes: displacements
            .iter()
            .filter(|displacement| norm(**displacement) > 0.0)
            .count(),
        max_displacement_m,
        min_element_measure_ratio,
        max_element_measure_ratio,
    })
}

/// Return the analytic nodal displacement Jacobian of the PC-fan ALE field.
///
/// The result is indexed by node and then by the public design-vector index;
/// each entry is `∂ displacement[node] / ∂ design[index]`. The FEM geometry
/// is rebuilt from `design`, so shroud-radius controls include their smooth
/// Catmull--Rom support derivative. Controls at the frame opening use the
/// active-set projection derivative of zero, matching the geometry clamp.
pub fn pc_fan_mesh_motion_design_jacobian(
    mesh: &Mesh,
    config: &PcFanConfig,
    design: &PcFanDesignVariables,
) -> Result<Vec<[[f64; 3]; DESIGN_VARIABLE_COUNT]>, PcFanMeshMotionError> {
    let geometry = config.build_geometry(design);
    let mut admitted_mesh = mesh.clone();
    apply_pc_fan_mesh_motion(&mut admitted_mesh, config, &geometry, design)?;
    mesh.validate_connectivity()
        .map_err(|_| PcFanMeshMotionError::InvalidMesh)?;

    let half_side = 0.5 * geometry.frame_box.outer_side_m + config.cfd.side_clearance_m.max(0.0);
    let fan_z0 = -0.5 * geometry.frame_box.depth_m;
    let fan_z1 = 0.5 * geometry.frame_box.depth_m;
    let cells_x = config.cfd.cells_x.max(1) as f64;
    let cells_z = config.cfd.cells_z.max(1) as f64;
    let radial_support = (2.0 * half_side / cells_x).max(MIN_SUPPORT_WIDTH_M);
    let axial_support = ((fan_z1 - fan_z0) / cells_z).max(MIN_SUPPORT_WIDTH_M);
    let hub_radius = design.effective_hub_radius_m(config);
    let tip_radius = design.effective_tip_radius_m(config);
    if !half_side.is_finite()
        || half_side <= 0.0
        || !fan_z0.is_finite()
        || !fan_z1.is_finite()
        || fan_z1 <= fan_z0
        || !hub_radius.is_finite()
        || !tip_radius.is_finite()
        || hub_radius <= 0.0
        || tip_radius <= hub_radius
    {
        return Err(PcFanMeshMotionError::InvalidInput("mesh-motion geometry"));
    }

    let max_inner_radius = (0.5 * config.dimensions.frame_side_m
        - config.dimensions.frame_wall_thickness_m
        - config.dimensions.shroud_wall_thickness_m)
        .max(0.0);
    let shroud_control_active = (0..SHROUD_CONTROL_COUNT)
        .map(|index| {
            let depth_fraction = station_fraction(index, SHROUD_CONTROL_COUNT);
            let raw_radius = config.dimensions.shroud_inner_radius_m
                + design.shroud_radius_offset_at(depth_fraction);
            if !raw_radius.is_finite() {
                return Err(PcFanMeshMotionError::InvalidInput(
                    "shroud geometry derivative",
                ));
            }
            Ok(raw_radius < max_inner_radius - 1.0e-12)
        })
        .collect::<Result<Vec<_>, PcFanMeshMotionError>>()?;

    let radial_start = BLADE_PATH_CONTROL_COUNT * 2 + BLADE_SECTION_STATION_COUNT * 4;
    let hub_index = radial_start;
    let tip_index = radial_start + 1;
    let shroud_start = radial_start + RADIAL_ENVELOPE_CONTROL_COUNT;
    let mut jacobian = vec![[[0.0; 3]; DESIGN_VARIABLE_COUNT]; mesh.nodes.len()];

    for (node_index, point) in mesh.nodes.iter().enumerate() {
        let radius = point.x.hypot(point.y);
        if radius <= MOTION_EPSILON_M {
            continue;
        }
        let radial_taper = smoothstep01((half_side - radius) / half_side);
        let axial_taper = smoothstep01((point.z - (fan_z0 - axial_support)) / axial_support)
            * smoothstep01(((fan_z1 + axial_support) - point.z) / axial_support);
        let depth_fraction = ((point.z - fan_z0) / (fan_z1 - fan_z0)).clamp(0.0, 1.0);
        let shroud_radius = geometry.shroud.radius_at_depth_fraction(depth_fraction);
        let shroud_basis = catmull_rom_basis(SHROUD_CONTROL_COUNT, depth_fraction);
        let shroud_radius_derivative = shroud_basis
            .iter()
            .copied()
            .zip(shroud_control_active.iter().copied())
            .map(|(basis, active)| if active { basis } else { 0.0 })
            .collect::<Vec<_>>();
        let shroud_offset = design.shroud_radius_offset_at(depth_fraction);
        let common = axial_taper * radial_taper;
        let hub_argument = (hub_radius + radial_support - radius) / radial_support;
        let tip_argument = (radius - tip_radius) / radial_support;
        let shroud_argument = (radius - shroud_radius) / radial_support;
        let hub_weight = smoothstep01(hub_argument);
        let tip_weight = bell_weight(tip_argument);
        let shroud_weight = bell_weight(shroud_argument);
        let hub_derivative = common
            * (hub_weight
                + design.hub_radius_offset_m * smoothstep_derivative01(hub_argument)
                    / radial_support);
        let tip_derivative = common
            * (tip_weight
                - design.tip_radius_offset_m * bell_weight_derivative(tip_argument)
                    / radial_support);
        let radial_direction = [point.x / radius, point.y / radius, 0.0];
        for component in 0..3 {
            jacobian[node_index][hub_index][component] =
                hub_derivative * radial_direction[component];
            jacobian[node_index][tip_index][component] =
                tip_derivative * radial_direction[component];
        }
        for (control, basis) in shroud_basis.into_iter().enumerate() {
            let shroud_radius_derivative = shroud_radius_derivative[control];
            let derivative = common
                * (basis * shroud_weight
                    - shroud_offset
                        * bell_weight_derivative(shroud_argument)
                        * shroud_radius_derivative
                        / radial_support);
            for component in 0..3 {
                jacobian[node_index][shroud_start + control][component] =
                    derivative * radial_direction[component];
            }
        }
    }
    Ok(jacobian)
}

fn smoothstep01(value: f64) -> f64 {
    let value = value.clamp(0.0, 1.0);
    value * value * (3.0 - 2.0 * value)
}

fn smoothstep_derivative01(value: f64) -> f64 {
    if !(0.0..1.0).contains(&value) {
        0.0
    } else {
        6.0 * value * (1.0 - value)
    }
}

fn bell_weight(value: f64) -> f64 {
    let distance = value.abs();
    if distance >= 1.0 {
        0.0
    } else {
        let complement = 1.0 - distance * distance;
        complement * complement
    }
}

fn bell_weight_derivative(value: f64) -> f64 {
    if value.abs() >= 1.0 {
        0.0
    } else {
        -4.0 * value * (1.0 - value * value)
    }
}

fn station_fraction(index: usize, count: usize) -> f64 {
    if count <= 1 {
        0.0
    } else {
        index as f64 / (count - 1) as f64
    }
}

fn catmull_rom_basis(count: usize, t: f64) -> Vec<f64> {
    let mut basis = vec![0.0; count];
    if count == 0 {
        return basis;
    }
    if count == 1 {
        basis[0] = 1.0;
        return basis;
    }
    let scaled = t.clamp(0.0, 1.0) * (count - 1) as f64;
    let i1 = scaled.floor() as usize;
    let i2 = (i1 + 1).min(count - 1);
    let i0 = i1.saturating_sub(1);
    let i3 = (i2 + 1).min(count - 1);
    let local_t = scaled - i1 as f64;
    let t2 = local_t * local_t;
    let t3 = t2 * local_t;
    let coefficients = [
        -0.5 * local_t + t2 - 0.5 * t3,
        1.0 - 2.5 * t2 + 1.5 * t3,
        0.5 * local_t + 2.0 * t2 - 1.5 * t3,
        -0.5 * t2 + 0.5 * t3,
    ];
    for (index, coefficient) in [i0, i1, i2, i3].into_iter().zip(coefficients) {
        basis[index] += coefficient;
    }
    basis
}

fn norm(value: [f64; 3]) -> f64 {
    value
        .iter()
        .map(|component| component * component)
        .sum::<f64>()
        .sqrt()
}

fn signed_tetra_measure(mesh: &Mesh, element_index: usize) -> Option<f64> {
    let nodes = &mesh.elements.get(element_index)?.nodes;
    if nodes.len() != 4 {
        return None;
    }
    let p0 = mesh.nodes.get(nodes[0])?;
    let p1 = mesh.nodes.get(nodes[1])?;
    let p2 = mesh.nodes.get(nodes[2])?;
    let p3 = mesh.nodes.get(nodes[3])?;
    let a = [p1.x - p0.x, p1.y - p0.y, p1.z - p0.z];
    let b = [p2.x - p0.x, p2.y - p0.y, p2.z - p0.z];
    let c = [p3.x - p0.x, p3.y - p0.y, p3.z - p0.z];
    Some(
        (a[0] * (b[1] * c[2] - b[2] * c[1]) - a[1] * (b[0] * c[2] - b[2] * c[0])
            + a[2] * (b[0] * c[1] - b[1] * c[0]))
            / 6.0,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::SHROUD_CONTROL_COUNT;
    use sonium_fem::mesh::generators::box_mesh_tetrahedra_symmetric;

    fn fixture() -> (PcFanConfig, PcFanGeometry, Mesh) {
        let mut config = PcFanConfig::default_120mm();
        config.cfd.cells_x = 3;
        config.cfd.cells_y = 3;
        config.cfd.cells_z = 3;
        let design = config.default_design_variables();
        let geometry = config.build_geometry(&design);
        let domain = geometry.build_cfd_domain(&config);
        let mesh = box_mesh_tetrahedra_symmetric(
            domain.min_corner.x,
            domain.max_corner.x,
            domain.min_corner.y,
            domain.max_corner.y,
            domain.min_corner.z,
            domain.max_corner.z,
            config.cfd.cells_x,
            config.cfd.cells_y,
            config.cfd.cells_z,
        );
        (config, geometry, mesh)
    }

    #[test]
    fn baseline_design_has_no_mesh_motion() {
        let (config, geometry, mut mesh) = fixture();
        let design = config.default_design_variables();
        let before = mesh.nodes.clone();
        let report = apply_pc_fan_mesh_motion(&mut mesh, &config, &geometry, &design)
            .expect("baseline motion should be admitted");
        assert!(!report.applied);
        assert_eq!(report.moved_nodes, 0);
        assert_eq!(mesh.nodes, before);
    }

    #[test]
    fn radial_and_shroud_offsets_move_mesh_transactionally() {
        let (config, geometry, mut mesh) = fixture();
        let mut design = config.default_design_variables();
        design.hub_radius_offset_m = 4.0e-4;
        design.tip_radius_offset_m = -3.0e-4;
        design.shroud_radius_offsets_m = [2.0e-4; SHROUD_CONTROL_COUNT];
        let before = mesh.nodes.clone();
        let before_elements = mesh.elements.clone();
        let report = apply_pc_fan_mesh_motion(&mut mesh, &config, &geometry, &design)
            .expect("bounded radial motion should be admitted");
        assert!(report.applied);
        assert!(report.moved_nodes > 0);
        assert!(report.max_displacement_m.is_finite());
        assert!(report.min_element_measure_ratio > 0.0);
        assert!(report.max_element_measure_ratio.is_finite());
        assert!(
            mesh.nodes
                .iter()
                .zip(before.iter())
                .any(|(after, before)| after != before)
        );
        assert_eq!(mesh.elements.len(), before_elements.len());
        for (after, before) in mesh.elements.iter().zip(before_elements.iter()) {
            assert_eq!(after.id, before.id);
            assert_eq!(after.nodes, before.nodes);
            assert_eq!(after.element_type, before.element_type);
        }
    }

    #[test]
    fn analytic_motion_jacobian_matches_centered_coordinate_motion() {
        let (config, _geometry, mesh) = fixture();
        let design = config.default_design_variables();
        let jacobian = pc_fan_mesh_motion_design_jacobian(&mesh, &config, &design)
            .expect("baseline mesh-motion Jacobian should be admitted");
        let radial_start = BLADE_PATH_CONTROL_COUNT * 2 + BLADE_SECTION_STATION_COUNT * 4;
        let cases = [(radial_start, 1.0e-5), (radial_start + 1, 1.0e-5)];

        for (index, step) in cases {
            let mut plus = design.clone();
            let mut minus = design.clone();
            let mut plus_vector = plus.to_vector();
            let mut minus_vector = minus.to_vector();
            plus_vector[index] += step;
            minus_vector[index] -= step;
            plus = PcFanDesignVariables::from_vector(plus_vector);
            minus = PcFanDesignVariables::from_vector(minus_vector);

            let mut plus_mesh = mesh.clone();
            let mut minus_mesh = mesh.clone();
            let plus_geometry = config.build_geometry(&plus);
            let minus_geometry = config.build_geometry(&minus);
            apply_pc_fan_mesh_motion(&mut plus_mesh, &config, &plus_geometry, &plus)
                .expect("positive perturbation should be admitted");
            apply_pc_fan_mesh_motion(&mut minus_mesh, &config, &minus_geometry, &minus)
                .expect("negative perturbation should be admitted");

            for node in 0..mesh.nodes.len() {
                for component in 0..3 {
                    let plus_value = match component {
                        0 => plus_mesh.nodes[node].x,
                        1 => plus_mesh.nodes[node].y,
                        _ => plus_mesh.nodes[node].z,
                    };
                    let minus_value = match component {
                        0 => minus_mesh.nodes[node].x,
                        1 => minus_mesh.nodes[node].y,
                        _ => minus_mesh.nodes[node].z,
                    };
                    let centered = (plus_value - minus_value) / (2.0 * step);
                    let analytic = jacobian[node][index][component];
                    assert!(
                        (analytic - centered).abs() <= 1.0e-4,
                        "node {node}, component {component}, design {index}: analytic {analytic} centered {centered}"
                    );
                }
            }
        }

        let mut shroud_design = design.clone();
        shroud_design.shroud_radius_offsets_m = [-1.0e-3; SHROUD_CONTROL_COUNT];
        let shroud_jacobian = pc_fan_mesh_motion_design_jacobian(&mesh, &config, &shroud_design)
            .expect("unclamped shroud Jacobian should be admitted");
        let shroud_index = radial_start + RADIAL_ENVELOPE_CONTROL_COUNT + 2;
        let step = 1.0e-6;
        let mut plus = shroud_design.clone();
        let mut minus = shroud_design.clone();
        plus.shroud_radius_offsets_m[2] += step;
        minus.shroud_radius_offsets_m[2] -= step;
        let mut plus_mesh = mesh.clone();
        let mut minus_mesh = mesh.clone();
        let plus_geometry = config.build_geometry(&plus);
        let minus_geometry = config.build_geometry(&minus);
        apply_pc_fan_mesh_motion(&mut plus_mesh, &config, &plus_geometry, &plus)
            .expect("positive shroud perturbation should be admitted");
        apply_pc_fan_mesh_motion(&mut minus_mesh, &config, &minus_geometry, &minus)
            .expect("negative shroud perturbation should be admitted");
        for node in 0..mesh.nodes.len() {
            for component in 0..3 {
                let plus_value = match component {
                    0 => plus_mesh.nodes[node].x,
                    1 => plus_mesh.nodes[node].y,
                    _ => plus_mesh.nodes[node].z,
                };
                let minus_value = match component {
                    0 => minus_mesh.nodes[node].x,
                    1 => minus_mesh.nodes[node].y,
                    _ => minus_mesh.nodes[node].z,
                };
                let centered = (plus_value - minus_value) / (2.0 * step);
                let analytic = shroud_jacobian[node][shroud_index][component];
                assert!((analytic - centered).abs() <= 1.0e-4);
            }
        }
    }

    #[test]
    fn motion_is_deterministic_and_preserves_outer_boundary() {
        let (config, geometry, mut first) = fixture();
        let mut second = first.clone();
        let mut design = config.default_design_variables();
        design.shroud_radius_offsets_m[1] = -2.5e-4;
        apply_pc_fan_mesh_motion(&mut first, &config, &geometry, &design).unwrap();
        apply_pc_fan_mesh_motion(&mut second, &config, &geometry, &design).unwrap();
        assert_eq!(first.nodes, second.nodes);
        for (node, original) in first.nodes.iter().zip(second.nodes.iter()) {
            assert!(node.x.is_finite() && node.y.is_finite() && node.z.is_finite());
            assert_eq!(node.z, original.z);
        }
    }

    #[test]
    fn non_finite_offset_fails_closed_without_mutating_mesh() {
        let (config, geometry, mut mesh) = fixture();
        let mut design = config.default_design_variables();
        design.tip_radius_offset_m = f64::NAN;
        let before = mesh.nodes.clone();
        assert!(matches!(
            apply_pc_fan_mesh_motion(&mut mesh, &config, &geometry, &design),
            Err(PcFanMeshMotionError::InvalidInput("design offsets"))
        ));
        assert_eq!(mesh.nodes, before);
    }

    #[test]
    fn excessive_motion_fails_quality_barrier_transactionally() {
        let (config, geometry, mut mesh) = fixture();
        let mut design = config.default_design_variables();
        design.shroud_radius_offsets_m = [0.2; SHROUD_CONTROL_COUNT];
        let before = mesh.nodes.clone();
        let result = apply_pc_fan_mesh_motion(&mut mesh, &config, &geometry, &design);
        assert!(
            matches!(
                result,
                Err(PcFanMeshMotionError::InvalidElements(_))
                    | Err(PcFanMeshMotionError::ElementMeasureCollapse { .. })
            ),
            "result={result:?}"
        );
        assert_eq!(mesh.nodes, before);
    }

    #[test]
    fn resolved_steady_solver_consumes_moved_mesh_coordinates() {
        let mut config = PcFanConfig::default_120mm();
        config.cfd.flow_model = crate::model::PcFanFlowModel::IncompressibleCfd;
        config.cfd.backend = crate::model::PcFanCfdBackend::Cpu;
        config.cfd.cells_x = 2;
        config.cfd.cells_y = 2;
        config.cfd.cells_z = 2;
        config.cfd.max_picard_iterations = 1;
        config.cfd.nonlinear_tolerance = 2.0;
        let baseline = config.default_design_variables();
        let mut moved = baseline.clone();
        moved.shroud_radius_offsets_m[1] = 1.0e-4;
        let baseline_domain = config.build_cfd_domain(&baseline);
        let baseline_fem = box_mesh_tetrahedra_symmetric(
            baseline_domain.min_corner.x,
            baseline_domain.max_corner.x,
            baseline_domain.min_corner.y,
            baseline_domain.max_corner.y,
            baseline_domain.min_corner.z,
            baseline_domain.max_corner.z,
            config.cfd.cells_x,
            config.cfd.cells_y,
            config.cfd.cells_z,
        );
        let solution = crate::cfd::solve_pc_fan_cfd_mean_flow(&config, &moved);
        assert_eq!(solution.nodes.len(), baseline_fem.nodes.len());
        assert!(
            solution
                .nodes
                .iter()
                .zip(baseline_fem.nodes.iter())
                .any(|(after, before)| {
                    (after.x - before.x).abs() > f64::EPSILON
                        || (after.y - before.y).abs() > f64::EPSILON
                        || (after.z - before.z).abs() > f64::EPSILON
                })
        );
    }

    #[test]
    fn rotating_wall_transient_admits_moved_mesh_design() {
        let mut config = PcFanConfig::default_120mm();
        config.cfd.flow_model = crate::model::PcFanFlowModel::IncompressibleCfdFull;
        config.cfd.backend = crate::model::PcFanCfdBackend::Cpu;
        config.cfd.cells_x = 2;
        config.cfd.cells_y = 2;
        config.cfd.cells_z = 2;
        config.cfd.max_picard_iterations = 1;
        config.cfd.nonlinear_tolerance = 2.0;
        config.cfd.max_relative_mass_imbalance = 10.0;
        config.cfd.max_relative_momentum_imbalance = 10.0;
        config.cfd.max_relative_pressure_closure = 10.0;
        config.cfd.max_relative_energy_imbalance = 10.0;
        let mut design = config.default_design_variables();
        design.tip_radius_offset_m = 1.0e-4;
        let trajectory = crate::cfd::solve_pc_fan_cfd_rotating_wall_transient_with_trajectory(
            &config, &design, 1.0e-3, 1.0e-3,
        )
        .expect("rotating-wall transient should admit bounded ALE motion");
        assert!(trajectory.observation.converged);
        assert!(trajectory.steps.len() >= 2);
    }
}
