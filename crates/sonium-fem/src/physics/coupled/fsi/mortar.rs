//! Nonmatching P1 dual-mortar operators for three-dimensional FSI interfaces.

use crate::assembly::TripletMatrix;
use crate::mesh::boundary_mesh::BoundarySubmesh;
use crate::physics::solid::contact::ContactSurface;
use crate::physics::solid::contact::detection::Aabb;
use crate::physics::solid::contact::gap::project_point_to_triangle;

use super::{AitkenRelaxation, FsiCouplingConfig, FsiError, FsiInterfaceConstraint};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FsiMortarMode {
    FullVector,
    NormalOnly,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FsiMortarSide {
    Fluid,
    Solid,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FsiMortarSideSelection {
    AutomaticCoarser,
    Explicit(FsiMortarSide),
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FsiMortarConfig {
    pub mode: FsiMortarMode,
    pub slave: FsiMortarSideSelection,
    pub maximum_relative_motion: f64,
    pub relative_projection_tolerance: f64,
    pub maximum_uncovered_area_fraction: f64,
    pub search_radius_factor: f64,
}

impl Default for FsiMortarConfig {
    fn default() -> Self {
        Self {
            mode: FsiMortarMode::FullVector,
            slave: FsiMortarSideSelection::AutomaticCoarser,
            maximum_relative_motion: 0.05,
            relative_projection_tolerance: 1.0e-8,
            maximum_uncovered_area_fraction: 1.0e-6,
            search_radius_factor: 2.0,
        }
    }
}

impl FsiMortarConfig {
    pub fn validate(self) -> Result<Self, FsiError> {
        if !self.maximum_relative_motion.is_finite()
            || self.maximum_relative_motion <= 0.0
            || !self.relative_projection_tolerance.is_finite()
            || self.relative_projection_tolerance <= 0.0
            || !self.maximum_uncovered_area_fraction.is_finite()
            || !(0.0..1.0).contains(&self.maximum_uncovered_area_fraction)
            || !self.search_radius_factor.is_finite()
            || self.search_radius_factor <= 0.0
        {
            return Err(FsiError::InvalidMortarConfig);
        }
        Ok(self)
    }
}

/// Triangulated P1 trace surface in local interface-node ordering.
#[derive(Debug, Clone, PartialEq)]
pub struct FsiMortarSurface {
    pub coordinates: Vec<[f64; 3]>,
    pub triangles: Vec<[usize; 3]>,
    pub parent_nodes: Vec<usize>,
}

impl FsiMortarSurface {
    pub fn new(
        coordinates: Vec<[f64; 3]>,
        triangles: Vec<[usize; 3]>,
        parent_nodes: Vec<usize>,
    ) -> Result<Self, FsiError> {
        if coordinates.is_empty()
            || triangles.is_empty()
            || parent_nodes.len() != coordinates.len()
            || triangles
                .iter()
                .flatten()
                .any(|&node| node >= coordinates.len())
        {
            return Err(FsiError::InvalidMortarSurface);
        }
        Ok(Self {
            coordinates,
            triangles,
            parent_nodes,
        })
    }

    /// Convert a boundary submesh to a triangle trace. Quads use the stable
    /// `(0,1,2)`/`(0,2,3)` diagonal.
    pub fn from_boundary_submesh(boundary: &BoundarySubmesh) -> Result<Self, FsiError> {
        let coordinates = boundary
            .mesh
            .nodes
            .iter()
            .map(|point| [point.x, point.y, point.z])
            .collect();
        let mut triangles = Vec::new();
        for element in &boundary.mesh.elements {
            match element.nodes.as_slice() {
                [a, b, c] => triangles.push([*a, *b, *c]),
                [a, b, c, d] => {
                    triangles.push([*a, *b, *c]);
                    triangles.push([*a, *c, *d]);
                }
                _ => return Err(FsiError::InvalidMortarSurface),
            }
        }
        Self::new(coordinates, triangles, boundary.node_to_parent.clone())
    }

    pub fn trace_dofs(&self) -> usize {
        3 * self.coordinates.len()
    }
}

#[derive(Debug)]
pub struct FsiMortarOperators {
    pub b_fluid: TripletMatrix<f64>,
    pub b_solid: TripletMatrix<f64>,
    pub fluid_trace_dofs: usize,
    pub solid_trace_dofs: usize,
    pub multiplier_dofs: usize,
    pub slave_side: FsiMortarSide,
    pub mode: FsiMortarMode,
    config: FsiMortarConfig,
    slave_reference_coordinates: Vec<[f64; 3]>,
    slave_node_scales: Vec<f64>,
}

impl FsiMortarOperators {
    pub fn build(
        fluid: &FsiMortarSurface,
        solid: &FsiMortarSurface,
        config: FsiMortarConfig,
    ) -> Result<Self, FsiError> {
        let config = config.validate()?;
        let slave_side = match config.slave {
            FsiMortarSideSelection::AutomaticCoarser => {
                if fluid.coordinates.len() <= solid.coordinates.len() {
                    FsiMortarSide::Fluid
                } else {
                    FsiMortarSide::Solid
                }
            }
            FsiMortarSideSelection::Explicit(side) => side,
        };
        let (slave, master) = match slave_side {
            FsiMortarSide::Fluid => (fluid, solid),
            FsiMortarSide::Solid => (solid, fluid),
        };
        let interface_scale = interface_diameter(&slave.coordinates);
        if !interface_scale.is_finite() || interface_scale <= f64::EPSILON {
            return Err(FsiError::InvalidMortarSurface);
        }
        let projection_tolerance = config.relative_projection_tolerance * interface_scale;
        let components = match config.mode {
            FsiMortarMode::FullVector => 3,
            FsiMortarMode::NormalOnly => 1,
        };
        let multiplier_dofs = components * slave.coordinates.len();
        let mut b_fluid = TripletMatrix::new(multiplier_dofs, fluid.trace_dofs());
        let mut b_solid = TripletMatrix::new(multiplier_dofs, solid.trace_dofs());

        let flat_master: Vec<f64> = master.coordinates.iter().flatten().copied().collect();
        let contact_surface =
            ContactSurface::from_boundary_faces(&master.triangles, master.coordinates.len());
        let tree = contact_surface.build_bvh(&flat_master, projection_tolerance);
        let slave_node_scales = incident_node_scales(slave);
        let total_area = slave
            .triangles
            .iter()
            .map(|triangle| triangle_area(slave, *triangle))
            .sum::<f64>();
        let mut uncovered_area = 0.0;

        // Degree-two triangle rule. It integrates the dual-P1/P1 products
        // exactly when a quadrature point remains on one master facet.
        const QUADRATURE: [[f64; 3]; 3] = [
            [2.0 / 3.0, 1.0 / 6.0, 1.0 / 6.0],
            [1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0],
            [1.0 / 6.0, 1.0 / 6.0, 2.0 / 3.0],
        ];

        for &slave_triangle in &slave.triangles {
            let area = triangle_area(slave, slave_triangle);
            if area <= f64::EPSILON {
                return Err(FsiError::InvalidMortarSurface);
            }
            let normal = triangle_normal(slave, slave_triangle);
            let face_scale = triangle_diameter(slave, slave_triangle);
            for barycentric in QUADRATURE {
                let point = interpolate_triangle(slave, slave_triangle, barycentric);
                let query_radius = config.search_radius_factor * face_scale;
                let query = Aabb {
                    min: [
                        point[0] - query_radius,
                        point[1] - query_radius,
                        point[2] - query_radius,
                    ],
                    max: [
                        point[0] + query_radius,
                        point[1] + query_radius,
                        point[2] + query_radius,
                    ],
                };
                let mut candidates = Vec::new();
                tree.query(&query, &mut candidates);
                let projection = candidates
                    .into_iter()
                    .map(|triangle_index| {
                        let triangle = master.triangles[triangle_index];
                        let result = project_point_to_triangle(
                            point,
                            master.coordinates[triangle[0]],
                            master.coordinates[triangle[1]],
                            master.coordinates[triangle[2]],
                        );
                        (result.distance, triangle, result)
                    })
                    .min_by(|left, right| left.0.total_cmp(&right.0));
                let Some((distance, master_triangle, projection)) = projection else {
                    uncovered_area += area / 3.0;
                    continue;
                };
                if distance > projection_tolerance {
                    uncovered_area += area / 3.0;
                    continue;
                }
                let master_basis = [
                    1.0 - projection.xi - projection.eta,
                    projection.xi,
                    projection.eta,
                ];
                let weight = area / 3.0;
                for local_multiplier in 0..3 {
                    let multiplier_node = slave_triangle[local_multiplier];
                    let dual_value = 4.0 * barycentric[local_multiplier] - 1.0;
                    match config.mode {
                        FsiMortarMode::FullVector => {
                            for component in 0..3 {
                                let row = 3 * multiplier_node + component;
                                add_trace_row(
                                    &mut b_fluid,
                                    &mut b_solid,
                                    slave_side,
                                    row,
                                    component,
                                    dual_value * weight,
                                    slave_triangle,
                                    barycentric,
                                    master_triangle,
                                    master_basis,
                                );
                            }
                        }
                        FsiMortarMode::NormalOnly => {
                            for (component, &n) in normal.iter().enumerate() {
                                add_trace_row(
                                    &mut b_fluid,
                                    &mut b_solid,
                                    slave_side,
                                    multiplier_node,
                                    component,
                                    dual_value * n * weight,
                                    slave_triangle,
                                    barycentric,
                                    master_triangle,
                                    master_basis,
                                );
                            }
                        }
                    }
                }
            }
        }

        if uncovered_area / total_area > config.maximum_uncovered_area_fraction {
            return Err(FsiError::MortarProjectionCoverage);
        }

        Ok(Self {
            b_fluid,
            b_solid,
            fluid_trace_dofs: fluid.trace_dofs(),
            solid_trace_dofs: solid.trace_dofs(),
            multiplier_dofs,
            slave_side,
            mode: config.mode,
            config,
            slave_reference_coordinates: slave.coordinates.clone(),
            slave_node_scales,
        })
    }

    pub fn constraint_residual(&self, fluid: &[f64], solid: &[f64]) -> Vec<f64> {
        assert_eq!(fluid.len(), self.fluid_trace_dofs);
        assert_eq!(solid.len(), self.solid_trace_dofs);
        let mut residual = vec![0.0; self.multiplier_dofs];
        apply_triplets(&self.b_fluid, fluid, 1.0, &mut residual);
        apply_triplets(&self.b_solid, solid, -1.0, &mut residual);
        residual
    }

    pub fn multiplier_forces(&self, multipliers: &[f64]) -> InterfaceVectorTransfer {
        assert_eq!(multipliers.len(), self.multiplier_dofs);
        let mut fluid_forces = vec![0.0; self.fluid_trace_dofs];
        let mut solid_forces = vec![0.0; self.solid_trace_dofs];
        apply_transpose_triplets(&self.b_fluid, multipliers, 1.0, &mut fluid_forces);
        apply_transpose_triplets(&self.b_solid, multipliers, -1.0, &mut solid_forces);
        InterfaceVectorTransfer {
            fluid_forces,
            solid_forces,
        }
    }

    pub fn assemble_saddle_operator(&self) -> FsiInterfaceConstraint {
        let total = self.fluid_trace_dofs + self.solid_trace_dofs + self.multiplier_dofs;
        let multiplier_offset = self.fluid_trace_dofs + self.solid_trace_dofs;
        let mut matrix = TripletMatrix::with_capacity(
            total,
            total,
            2 * (self.b_fluid.nnz() + self.b_solid.nnz()),
        );
        for k in 0..self.b_fluid.nnz() {
            let row = self.b_fluid.rows[k];
            let column = self.b_fluid.cols[k];
            let value = self.b_fluid.values[k];
            matrix.push(column, multiplier_offset + row, value);
            matrix.push(multiplier_offset + row, column, value);
        }
        for k in 0..self.b_solid.nnz() {
            let row = self.b_solid.rows[k];
            let column = self.b_solid.cols[k] + self.fluid_trace_dofs;
            let value = -self.b_solid.values[k];
            matrix.push(column, multiplier_offset + row, value);
            matrix.push(multiplier_offset + row, column, value);
        }
        FsiInterfaceConstraint {
            matrix,
            fluid_dofs: self.fluid_trace_dofs,
            solid_dofs: self.solid_trace_dofs,
            multiplier_dofs: self.multiplier_dofs,
        }
    }

    /// Test the configured 5%-by-default reference-map rebuild criterion.
    pub fn needs_rebuild(
        &self,
        fluid_coordinates: &[[f64; 3]],
        solid_coordinates: &[[f64; 3]],
    ) -> Result<bool, FsiError> {
        let current = match self.slave_side {
            FsiMortarSide::Fluid => fluid_coordinates,
            FsiMortarSide::Solid => solid_coordinates,
        };
        if current.len() != self.slave_reference_coordinates.len() {
            return Err(FsiError::InvalidMortarSurface);
        }
        Ok(current.iter().enumerate().any(|(node, coordinate)| {
            distance(*coordinate, self.slave_reference_coordinates[node])
                > self.config.maximum_relative_motion * self.slave_node_scales[node]
        }))
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct InterfaceVectorTransfer {
    pub fluid_forces: Vec<f64>,
    pub solid_forces: Vec<f64>,
}

impl InterfaceVectorTransfer {
    pub fn virtual_work(&self, fluid: &[f64], solid: &[f64]) -> f64 {
        dot(&self.fluid_forces, fluid) + dot(&self.solid_forces, solid)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct FsiPartitionedReport {
    pub interface_state: Vec<f64>,
    pub iterations: usize,
    pub relative_residual: f64,
}

/// Callback-based Dirichlet-Neumann fixed-point iteration with Aitken
/// relaxation. Callbacks own their time integrators and state rollback.
pub struct FsiPartitionedCoupler {
    pub config: FsiCouplingConfig,
}

impl FsiPartitionedCoupler {
    pub fn new(config: FsiCouplingConfig) -> Result<Self, FsiError> {
        Ok(Self {
            config: config.validate()?,
        })
    }

    pub fn iterate<Fluid, Solid>(
        &self,
        initial_interface_state: &[f64],
        mut advance_fluid: Fluid,
        mut advance_solid: Solid,
    ) -> Result<FsiPartitionedReport, FsiError>
    where
        Fluid: FnMut(&[f64]) -> Result<Vec<f64>, FsiError>,
        Solid: FnMut(&[f64]) -> Result<Vec<f64>, FsiError>,
    {
        let mut state = initial_interface_state.to_vec();
        let mut aitken = AitkenRelaxation::new(
            self.config.initial_relaxation,
            self.config.minimum_relaxation,
            self.config.maximum_relaxation,
        );
        for iteration in 1..=self.config.maximum_iterations {
            let traction = advance_fluid(&state)?;
            let candidate = advance_solid(&traction)?;
            if candidate.len() != state.len() {
                return Err(FsiError::InterfaceSizeMismatch);
            }
            let residual = candidate
                .iter()
                .zip(&state)
                .map(|(new, old)| new - old)
                .collect::<Vec<_>>();
            let relative_residual = norm(&residual) / norm(&candidate).max(1.0);
            state = aitken.update(&state, &candidate);
            if relative_residual <= self.config.interface_tolerance {
                return Ok(FsiPartitionedReport {
                    interface_state: state,
                    iterations: iteration,
                    relative_residual,
                });
            }
        }
        Err(FsiError::InterfaceIterationsDidNotConverge)
    }
}

#[expect(
    clippy::too_many_arguments,
    reason = "triplet-assembly kernel threading matrix, maps and local contributions; grouping would churn hot assembly call sites"
)]
fn add_trace_row(
    b_fluid: &mut TripletMatrix<f64>,
    b_solid: &mut TripletMatrix<f64>,
    slave_side: FsiMortarSide,
    row: usize,
    component: usize,
    scale: f64,
    slave_triangle: [usize; 3],
    slave_basis: [f64; 3],
    master_triangle: [usize; 3],
    master_basis: [f64; 3],
) {
    let (slave_matrix, master_matrix) = match slave_side {
        FsiMortarSide::Fluid => (b_fluid, b_solid),
        FsiMortarSide::Solid => (b_solid, b_fluid),
    };
    for local in 0..3 {
        slave_matrix.push(
            row,
            3 * slave_triangle[local] + component,
            scale * slave_basis[local],
        );
        master_matrix.push(
            row,
            3 * master_triangle[local] + component,
            scale * master_basis[local],
        );
    }
}

fn apply_triplets(matrix: &TripletMatrix<f64>, x: &[f64], scale: f64, y: &mut [f64]) {
    for k in 0..matrix.nnz() {
        y[matrix.rows[k]] += scale * matrix.values[k] * x[matrix.cols[k]];
    }
}

fn apply_transpose_triplets(matrix: &TripletMatrix<f64>, x: &[f64], scale: f64, y: &mut [f64]) {
    for k in 0..matrix.nnz() {
        y[matrix.cols[k]] += scale * matrix.values[k] * x[matrix.rows[k]];
    }
}

fn incident_node_scales(surface: &FsiMortarSurface) -> Vec<f64> {
    let mut scales = vec![f64::INFINITY; surface.coordinates.len()];
    for &triangle in &surface.triangles {
        let diameter = triangle_diameter(surface, triangle);
        for node in triangle {
            scales[node] = scales[node].min(diameter);
        }
    }
    scales
}

fn interface_diameter(coordinates: &[[f64; 3]]) -> f64 {
    let bbox = Aabb::from_points(coordinates);
    distance(bbox.min, bbox.max)
}

fn triangle_area(surface: &FsiMortarSurface, triangle: [usize; 3]) -> f64 {
    let a = surface.coordinates[triangle[0]];
    let b = surface.coordinates[triangle[1]];
    let c = surface.coordinates[triangle[2]];
    0.5 * norm3(cross3(sub3(b, a), sub3(c, a)))
}

fn triangle_diameter(surface: &FsiMortarSurface, triangle: [usize; 3]) -> f64 {
    let a = surface.coordinates[triangle[0]];
    let b = surface.coordinates[triangle[1]];
    let c = surface.coordinates[triangle[2]];
    distance(a, b).max(distance(b, c)).max(distance(c, a))
}

fn triangle_normal(surface: &FsiMortarSurface, triangle: [usize; 3]) -> [f64; 3] {
    let raw = cross3(
        sub3(
            surface.coordinates[triangle[1]],
            surface.coordinates[triangle[0]],
        ),
        sub3(
            surface.coordinates[triangle[2]],
            surface.coordinates[triangle[0]],
        ),
    );
    let length = norm3(raw);
    [raw[0] / length, raw[1] / length, raw[2] / length]
}

fn interpolate_triangle(
    surface: &FsiMortarSurface,
    triangle: [usize; 3],
    basis: [f64; 3],
) -> [f64; 3] {
    let mut point = [0.0; 3];
    for local in 0..3 {
        for (component, slot) in point.iter_mut().enumerate() {
            *slot += basis[local] * surface.coordinates[triangle[local]][component];
        }
    }
    point
}

fn sub3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn cross3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn norm3(a: [f64; 3]) -> f64 {
    (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt()
}

fn distance(a: [f64; 3], b: [f64; 3]) -> f64 {
    norm3(sub3(a, b))
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(left, right)| left * right).sum()
}

fn norm(values: &[f64]) -> f64 {
    dot(values, values).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn square_surface(refined: bool) -> FsiMortarSurface {
        if refined {
            FsiMortarSurface::new(
                vec![
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [1.0, 1.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.5, 0.5, 0.0],
                ],
                vec![[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]],
                vec![0, 1, 2, 3, 4],
            )
            .unwrap()
        } else {
            FsiMortarSurface::new(
                vec![
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [1.0, 1.0, 0.0],
                    [0.0, 1.0, 0.0],
                ],
                vec![[0, 1, 2], [0, 2, 3]],
                vec![0, 1, 2, 3],
            )
            .unwrap()
        }
    }

    #[test]
    fn nonmatching_mortar_passes_constant_vector_patch() {
        let fluid = square_surface(false);
        let solid = square_surface(true);
        let operators =
            FsiMortarOperators::build(&fluid, &solid, FsiMortarConfig::default()).unwrap();
        let fluid_values = vec![2.5; fluid.trace_dofs()];
        let solid_values = vec![2.5; solid.trace_dofs()];
        let residual = operators.constraint_residual(&fluid_values, &solid_values);
        assert!(norm(&residual) < 1.0e-12, "residual={residual:?}");
    }

    #[test]
    fn multiplier_forces_preserve_virtual_work() {
        let fluid = square_surface(false);
        let solid = square_surface(true);
        let operators =
            FsiMortarOperators::build(&fluid, &solid, FsiMortarConfig::default()).unwrap();
        let fluid_values = vec![1.25; fluid.trace_dofs()];
        let solid_values = vec![1.25; solid.trace_dofs()];
        let multipliers = vec![0.75; operators.multiplier_dofs];
        let transfer = operators.multiplier_forces(&multipliers);
        assert!(transfer.virtual_work(&fluid_values, &solid_values).abs() < 1.0e-12);
    }

    #[test]
    fn reference_map_rebuilds_above_five_percent_face_scale() {
        let fluid = square_surface(false);
        let solid = square_surface(true);
        let operators =
            FsiMortarOperators::build(&fluid, &solid, FsiMortarConfig::default()).unwrap();
        let mut moved = fluid.coordinates.clone();
        moved[0][2] += 0.08;
        assert!(operators.needs_rebuild(&moved, &solid.coordinates).unwrap());
        moved[0][2] = 0.01;
        assert!(!operators.needs_rebuild(&moved, &solid.coordinates).unwrap());
    }

    #[test]
    fn callback_coupler_converges_fixed_point() {
        let coupler = FsiPartitionedCoupler::new(FsiCouplingConfig::default()).unwrap();
        let report = coupler
            .iterate(
                &[0.0],
                |state| Ok(state.to_vec()),
                |traction| Ok(vec![0.5 * traction[0] + 1.0]),
            )
            .unwrap();
        assert!((report.interface_state[0] - 2.0).abs() < 1.0e-7);
    }
}
