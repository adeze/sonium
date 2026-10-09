//! Three-dimensional cyclic motor sector with an H(curl)-H1 A-phi solve.

use crate::conventions::StudyMetadata;
use crate::field_export::{
    ExportedSnapshot, FieldExportError, ProjectedCellVector, ProjectionProvenance,
    SpeakerFieldBundle, SpeakerFieldExporter, SpeakerPvdCollection, field_names,
};
use num_complex::Complex64;
use sonium_fem::boundary::{MpcError, MpcSystem, MultiPointConstraint, SectorPeriodicMap};
use sonium_fem::mesh::{BoundaryType, ElementType, Mesh, PhysicalGroupError, Point};
use sonium_fem::physics::electromagnetics::eddy_current::{
    AphiAssembly, AphiError, AphiFieldSummary, AphiMaterial, AxisymmetricEddyCurrentProblem,
    DifferentialReluctivityField, TreeCotreeGauge, apply_zero_constraints_csr, assemble_aphi,
    evaluate_aphi_fields,
};
use sonium_fem::solver::{SolverConfig, SolverError, SolverType, solve_csr};
use sonium_fem::spaces::{FunctionSpace, HcurlSpace};
use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::f64::consts::PI;
use std::path::Path;

pub const SECTOR_AIR: i32 = 70;
pub const SECTOR_CONDUCTOR: i32 = 71;
pub const SECTOR_MASTER_FACE: i32 = 170;
pub const SECTOR_SLAVE_FACE: i32 = 171;
pub const SECTOR_INNER_RADIAL: i32 = 172;
pub const SECTOR_OUTER_RADIAL: i32 = 173;
pub const SECTOR_AXIAL_ENDS: i32 = 174;

#[derive(Debug, Clone)]
pub struct Sector3dConfig {
    pub sector_angle_degrees: f64,
    pub circumferential_order: i32,
    pub inner_radius_m: f64,
    pub conductor_inner_radius_m: f64,
    pub conductor_outer_radius_m: f64,
    pub outer_radius_m: f64,
    pub axial_half_height_m: f64,
    pub conductor_axial_half_height_m: f64,
    pub radial_cells_per_region: usize,
    pub axial_cells_per_region: usize,
    pub circumferential_cells: usize,
    pub conductor_relative_permeability: f64,
    pub conductor_conductivity_s_per_m: f64,
    pub current_rms_a: f64,
    pub quadrature_order: usize,
    pub periodic_tolerance_m: f64,
}

impl Default for Sector3dConfig {
    fn default() -> Self {
        Self {
            sector_angle_degrees: 30.0,
            circumferential_order: 0,
            inner_radius_m: 20.0e-3,
            conductor_inner_radius_m: 30.0e-3,
            conductor_outer_radius_m: 50.0e-3,
            outer_radius_m: 60.0e-3,
            axial_half_height_m: 20.0e-3,
            conductor_axial_half_height_m: 10.0e-3,
            radial_cells_per_region: 1,
            axial_cells_per_region: 1,
            circumferential_cells: 2,
            conductor_relative_permeability: 1.0,
            conductor_conductivity_s_per_m: 6.0e6,
            current_rms_a: 1.0,
            quadrature_order: 2,
            periodic_tolerance_m: 1.0e-10,
        }
    }
}

impl Sector3dConfig {
    pub fn validate(&self) -> Result<(), Sector3dError> {
        let positive = [
            self.sector_angle_degrees,
            self.inner_radius_m,
            self.conductor_inner_radius_m,
            self.conductor_outer_radius_m,
            self.outer_radius_m,
            self.axial_half_height_m,
            self.conductor_axial_half_height_m,
            self.conductor_relative_permeability,
            self.conductor_conductivity_s_per_m,
            self.current_rms_a,
            self.periodic_tolerance_m,
        ];
        if positive
            .iter()
            .any(|value| !value.is_finite() || *value <= 0.0)
        {
            return Err(Sector3dError::InvalidConfiguration(
                "dimensions, material values, angle, current, and tolerance must be positive",
            ));
        }
        if !crate::is_proper_divisor_of_360(self.sector_angle_degrees) {
            return Err(Sector3dError::InvalidConfiguration(
                "sector angle must be a proper divisor of 360 degrees",
            ));
        }
        if !(self.inner_radius_m < self.conductor_inner_radius_m
            && self.conductor_inner_radius_m < self.conductor_outer_radius_m
            && self.conductor_outer_radius_m < self.outer_radius_m
            && self.conductor_axial_half_height_m < self.axial_half_height_m)
        {
            return Err(Sector3dError::InvalidConfiguration(
                "conductor must lie strictly inside the annular magnetic domain",
            ));
        }
        if self.radial_cells_per_region == 0
            || self.axial_cells_per_region == 0
            || self.circumferential_cells < 2
            || self.quadrature_order == 0
        {
            return Err(Sector3dError::InvalidConfiguration(
                "mesh regions need cells, including two or more circumferential layers",
            ));
        }
        Ok(())
    }

    pub fn sector_angle_rad(&self) -> f64 {
        self.sector_angle_degrees.to_radians()
    }

    pub fn conductor_cross_section_area_m2(&self) -> f64 {
        (self.conductor_outer_radius_m - self.conductor_inner_radius_m)
            * 2.0
            * self.conductor_axial_half_height_m
    }

    pub fn source_current_density_per_amp_a_per_m2(&self) -> f64 {
        1.0 / self.conductor_cross_section_area_m2()
    }
}

pub struct Sector3dModel {
    pub config: Sector3dConfig,
    pub mesh: Mesh,
    pub periodic_map: SectorPeriodicMap,
    pub materials: BTreeMap<i32, AphiMaterial>,
}

#[derive(Debug, Clone)]
pub struct Sector3dSolution {
    pub frequency_hz: f64,
    pub vector_potential_edge_dofs: Vec<Complex64>,
    pub electric_potential_node_dofs: Vec<Complex64>,
    pub fields: AphiFieldSummary,
    pub full_revolution_eddy_loss_w: f64,
    pub full_revolution_magnetic_energy_j: f64,
    pub full_revolution_flux_linkage_wb_turn: Complex64,
    pub magnetic_impedance_ohm: Complex64,
    pub periodic_residual_max: f64,
    pub gauge_residual_max: f64,
    pub charge_continuity_relative_residual: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AxisymmetricSectorComparison {
    pub sector_impedance_ohm: Complex64,
    pub axisymmetric_impedance_ohm: Complex64,
    pub impedance_relative_error: f64,
    pub sector_eddy_loss_w: f64,
    pub axisymmetric_eddy_loss_w: f64,
    pub eddy_loss_relative_error: f64,
}

impl Sector3dModel {
    pub fn build(config: Sector3dConfig) -> Result<Self, Sector3dError> {
        config.validate()?;
        let mesh = build_sector_mesh(&config)?;
        let periodic_map = SectorPeriodicMap::from_marked_faces(
            &mesh,
            SECTOR_MASTER_FACE,
            SECTOR_SLAVE_FACE,
            config.sector_angle_rad(),
            config.circumferential_order,
            config.periodic_tolerance_m,
        )?;
        let materials = BTreeMap::from([
            (
                SECTOR_AIR,
                AphiMaterial::from_relative_permeability(1.0, 0.0),
            ),
            (
                SECTOR_CONDUCTOR,
                AphiMaterial::from_relative_permeability(
                    config.conductor_relative_permeability,
                    config.conductor_conductivity_s_per_m,
                ),
            ),
        ]);
        Ok(Self {
            config,
            mesh,
            periodic_map,
            materials,
        })
    }

    pub fn solve_frequency(&self, frequency_hz: f64) -> Result<Sector3dSolution, Sector3dError> {
        let current_density_per_amp = self.config.source_current_density_per_amp_a_per_m2();
        let source_per_amp = |marker, point| {
            azimuthal_source(
                marker,
                point,
                SECTOR_CONDUCTOR,
                current_density_per_amp,
                self.config.circumferential_order,
            )
        };
        let source_actual = |marker, point| {
            azimuthal_source(
                marker,
                point,
                SECTOR_CONDUCTOR,
                current_density_per_amp * self.config.current_rms_a,
                self.config.circumferential_order,
            )
        };
        let assembly = assemble_aphi(
            &self.mesh,
            &self.materials,
            frequency_hz,
            self.config.quadrature_order,
            source_actual,
        )?;
        let constraints = self.periodic_constraints(&assembly)?;
        let mpc = MpcSystem::new(assembly.total_dofs(), &constraints)?;

        let hcurl = HcurlSpace::new();
        let mut essential = hcurl.boundary_dofs(&self.mesh, SECTOR_INNER_RADIAL);
        essential.extend(hcurl.boundary_dofs(&self.mesh, SECTOR_OUTER_RADIAL));
        essential.extend(hcurl.boundary_dofs(&self.mesh, SECTOR_AXIAL_ENDS));
        let slave_edges: HashSet<_> = self
            .periodic_map
            .edge_pairs
            .iter()
            .map(|pair| pair.1)
            .collect();
        let gauge = TreeCotreeGauge::from_mesh_excluding(&self.mesh, &slave_edges)?;
        essential.extend(gauge.tree_edge_dofs.iter().copied());
        essential.extend(assembly.inactive_phi_dofs());
        let scalar_slave_nodes: HashSet<_> = self
            .periodic_map
            .node_pairs
            .iter()
            .map(|pair| pair.1)
            .collect();
        let phi_reference = assembly
            .active_phi_nodes
            .iter()
            .copied()
            // HashSet iteration is intentionally randomized.  Choosing the
            // first available node made the scalar-potential gauge depend on
            // the allocator/hash seed, so two otherwise identical solves
            // could publish different nodal potentials even when all
            // gauge-invariant fields and impedances agreed.  Pin the lowest
            // admissible master node to make the replay gauge deterministic.
            .filter(|node| !scalar_slave_nodes.contains(node))
            .min()
            .ok_or(Sector3dError::MissingScalarReference)?;
        essential.push(assembly.n_a_dofs + phi_reference);
        essential.sort_unstable();
        essential.dedup();
        let reduced_matrix = mpc.condense_matrix(&assembly.matrix.to_csr())?;
        let mut reduced_rhs = mpc.condense_rhs(&assembly.rhs)?;
        let physical_reduced_matrix = reduced_matrix.clone();
        let physical_reduced_rhs = reduced_rhs.clone();
        // Essential conditions must be applied after MPC condensation. Applying
        // an identity row on a full master first would still receive physical
        // contributions from its dependent slave through P^H A P.
        let reduced_essential: BTreeSet<_> = essential
            .iter()
            .flat_map(|dof| {
                mpc.prolongation_row(*dof)
                    .iter()
                    .filter_map(|(reduced, coefficient)| {
                        (coefficient.norm() > 0.0).then_some(*reduced)
                    })
            })
            .collect();
        let free_phi_rows: BTreeSet<_> = assembly
            .active_phi_nodes
            .iter()
            .flat_map(|node| mpc.prolongation_row(assembly.n_a_dofs + *node))
            .filter_map(|(reduced, coefficient)| {
                (coefficient.norm() > 0.0 && !reduced_essential.contains(reduced))
                    .then_some(*reduced)
            })
            .collect();
        let reduced_matrix = apply_zero_constraints_csr(
            &reduced_matrix,
            &mut reduced_rhs,
            &reduced_essential.into_iter().collect::<Vec<_>>(),
        )?;
        let solver = SolverConfig {
            solver_type: SolverType::Direct,
            ..SolverConfig::default()
        };
        let reduced = solve_csr(&reduced_matrix, &reduced_rhs, &solver)?;
        let source_scale = physical_reduced_rhs
            .iter()
            .map(|value| value.norm())
            .fold(0.0, f64::max)
            .max(1.0e-30);
        let charge_continuity_relative_residual = free_phi_rows
            .iter()
            .map(|row| {
                let applied: Complex64 = physical_reduced_matrix
                    .row_entries(*row)
                    .map(|(column, value)| value * reduced.values[column])
                    .sum();
                (applied - physical_reduced_rhs[*row]).norm() / source_scale
            })
            .fold(0.0, f64::max);
        let solution = mpc.recover(&reduced.values)?;
        if solution.iter().any(|value| !complex_is_finite(*value)) {
            return Err(Sector3dError::NonFiniteSolution);
        }
        let fields = evaluate_aphi_fields(
            &self.mesh,
            &self.materials,
            frequency_hz,
            &solution,
            source_per_amp,
        )?;
        let revolution_scale = 2.0 * PI / self.config.sector_angle_rad();
        let full_loss = revolution_scale * fields.eddy_loss_w;
        let full_energy = revolution_scale * fields.magnetic_energy_j;
        let full_flux = revolution_scale * fields.source_flux_linkage_wb_turn;
        let omega = 2.0 * PI * frequency_hz;
        let impedance = Complex64::new(0.0, omega) * full_flux / self.config.current_rms_a;
        let periodic_residual_max = self
            .periodic_map
            .hcurl_constraints()?
            .iter()
            .map(|constraint| constraint.residual(&solution).norm())
            .fold(0.0, f64::max);
        let gauge_residual_max = gauge
            .tree_edge_dofs
            .iter()
            .map(|dof| solution[*dof].norm())
            .fold(0.0, f64::max);
        let n_a = assembly.n_a_dofs;
        Ok(Sector3dSolution {
            frequency_hz,
            vector_potential_edge_dofs: solution[..n_a].to_vec(),
            electric_potential_node_dofs: solution[n_a..].to_vec(),
            fields,
            full_revolution_eddy_loss_w: full_loss,
            full_revolution_magnetic_energy_j: full_energy,
            full_revolution_flux_linkage_wb_turn: full_flux,
            magnetic_impedance_ohm: impedance,
            periodic_residual_max,
            gauge_residual_max,
            charge_continuity_relative_residual,
        })
    }

    pub fn compare_m0_axisymmetric(
        &self,
        frequency_hz: f64,
    ) -> Result<AxisymmetricSectorComparison, Sector3dError> {
        if self.config.circumferential_order != 0 {
            return Err(Sector3dError::ComparisonRequiresM0);
        }
        let sector = self.solve_frequency(frequency_hz)?;
        let axisymmetric = self.solve_axisymmetric_reference(frequency_hz)?;
        let impedance_relative_error =
            (sector.magnetic_impedance_ohm - axisymmetric.magnetic_impedance_ohm).norm()
                / axisymmetric.magnetic_impedance_ohm.norm().max(1.0e-30);
        let eddy_loss_relative_error =
            (sector.full_revolution_eddy_loss_w - axisymmetric.eddy_loss_w).abs()
                / axisymmetric.eddy_loss_w.abs().max(1.0e-30);
        Ok(AxisymmetricSectorComparison {
            sector_impedance_ohm: sector.magnetic_impedance_ohm,
            axisymmetric_impedance_ohm: axisymmetric.magnetic_impedance_ohm,
            impedance_relative_error,
            sector_eddy_loss_w: sector.full_revolution_eddy_loss_w,
            axisymmetric_eddy_loss_w: axisymmetric.eddy_loss_w,
            eddy_loss_relative_error,
        })
    }

    pub fn export_solution(
        &self,
        solution: &Sector3dSolution,
        path: impl AsRef<Path>,
    ) -> Result<ExportedSnapshot, Sector3dError> {
        let fields = self.field_bundle(solution)?;
        let mut metadata = StudyMetadata::new("3D periodic A-phi motor sector");
        metadata.frequency_hz = Some(solution.frequency_hz);
        metadata.sector_angle_degrees = Some(self.config.sector_angle_degrees);
        metadata.circumferential_mode = self.config.circumferential_order;
        metadata.reference_case = Some("axisymmetric A-theta m=0".into());
        Ok(SpeakerFieldExporter::new(&self.mesh, metadata, &fields).write_snapshot(path)?)
    }

    pub fn export_sweep(
        &self,
        frequencies_hz: &[f64],
        pvd_path: impl AsRef<Path>,
    ) -> Result<Vec<Sector3dSolution>, Sector3dError> {
        let mut collection = SpeakerPvdCollection::new(pvd_path)?;
        let mut solutions = Vec::with_capacity(frequencies_hz.len());
        for &frequency in frequencies_hz {
            let solution = self.solve_frequency(frequency)?;
            let fields = self.field_bundle(&solution)?;
            let mut metadata = StudyMetadata::new("3D periodic A-phi motor sector");
            metadata.frequency_hz = Some(frequency);
            metadata.sector_angle_degrees = Some(self.config.sector_angle_degrees);
            metadata.circumferential_mode = self.config.circumferential_order;
            metadata.reference_case = Some("axisymmetric A-theta m=0".into());
            collection.push(&SpeakerFieldExporter::new(&self.mesh, metadata, &fields))?;
            solutions.push(solution);
        }
        collection.finish()?;
        Ok(solutions)
    }

    fn periodic_constraints(
        &self,
        assembly: &AphiAssembly,
    ) -> Result<Vec<MultiPointConstraint>, Sector3dError> {
        let mut constraints: Vec<_> = self
            .periodic_map
            .hcurl_constraints()?
            .into_iter()
            .map(Into::into)
            .collect();
        constraints.extend(
            self.periodic_map
                .scalar_constraints(assembly.n_a_dofs, Some(&assembly.active_phi_nodes))?,
        );
        Ok(constraints)
    }

    fn solve_axisymmetric_reference(
        &self,
        frequency_hz: f64,
    ) -> Result<
        sonium_fem::physics::electromagnetics::eddy_current::AxisymmetricEddyCurrentSolution,
        Sector3dError,
    > {
        let mesh = build_axisymmetric_reference_mesh(&self.config);
        let reluctivity = DifferentialReluctivityField {
            by_element_m_per_h: mesh
                .elements
                .iter()
                .map(|element| {
                    let material = self.materials[&element.marker];
                    1.0 / material.permeability_h_per_m
                })
                .collect(),
        };
        let problem = AxisymmetricEddyCurrentProblem {
            mesh,
            differential_reluctivity: reluctivity,
            conductivity_s_per_m: BTreeMap::from([(
                SECTOR_CONDUCTOR,
                self.config.conductor_conductivity_s_per_m,
            )]),
            source_current_density_per_amp_a_per_m2: BTreeMap::from([(
                SECTOR_CONDUCTOR,
                self.config.source_current_density_per_amp_a_per_m2(),
            )]),
            quadrature_order: self.config.quadrature_order,
        };
        let tolerance = 1.0e-12;
        let boundary: Vec<_> = problem
            .mesh
            .nodes
            .iter()
            .enumerate()
            .filter_map(|(node, point)| {
                ((point.x - self.config.inner_radius_m).abs() < tolerance
                    || (point.x - self.config.outer_radius_m).abs() < tolerance
                    || (point.y - self.config.axial_half_height_m).abs() < tolerance
                    || (point.y + self.config.axial_half_height_m).abs() < tolerance)
                    .then_some(node)
            })
            .collect();
        Ok(problem.solve_unit_current(frequency_hz, &boundary)?)
    }

    fn field_bundle(
        &self,
        solution: &Sector3dSolution,
    ) -> Result<SpeakerFieldBundle, FieldExportError> {
        let mut fields = SpeakerFieldBundle::new();
        fields.add_point_complex_scalar(
            "electric_scalar_potential",
            solution.electric_potential_node_dofs.clone(),
        )?;
        fields.add_cell_complex_vector(
            field_names::MAGNETIC_POTENTIAL,
            ProjectedCellVector::new(
                solution
                    .fields
                    .cells
                    .iter()
                    .map(|cell| cell.vector_potential_t_m)
                    .collect(),
                ProjectionProvenance::ElementCentroid,
            ),
        )?;
        fields.add_cell_complex_vector(
            field_names::MAGNETIC_FLUX_DENSITY,
            ProjectedCellVector::new(
                solution
                    .fields
                    .cells
                    .iter()
                    .map(|cell| cell.flux_density_t)
                    .collect(),
                ProjectionProvenance::ElementCentroid,
            ),
        )?;
        fields.add_cell_complex_vector(
            "electric_field",
            ProjectedCellVector::new(
                solution
                    .fields
                    .cells
                    .iter()
                    .map(|cell| cell.electric_field_v_per_m)
                    .collect(),
                ProjectionProvenance::ElementCentroid,
            ),
        )?;
        fields.add_cell_complex_vector(
            field_names::EDDY_CURRENT_DENSITY,
            ProjectedCellVector::new(
                solution
                    .fields
                    .cells
                    .iter()
                    .map(|cell| cell.current_density_a_per_m2)
                    .collect(),
                ProjectionProvenance::ElementCentroid,
            ),
        )?;
        fields.add_cell_real_scalar(
            field_names::EDDY_LOSS_DENSITY,
            solution
                .fields
                .cells
                .iter()
                .map(|cell| cell.eddy_loss_density_w_per_m3)
                .collect(),
        )?;
        fields.add_cell_real_scalar(
            field_names::REGION_MARKER,
            self.mesh
                .elements
                .iter()
                .map(|element| element.marker as f64)
                .collect(),
        )?;
        Ok(fields)
    }
}

fn build_sector_mesh(config: &Sector3dConfig) -> Result<Mesh, PhysicalGroupError> {
    let radial = segmented_coordinates(&[
        (
            config.inner_radius_m,
            config.conductor_inner_radius_m,
            config.radial_cells_per_region,
        ),
        (
            config.conductor_inner_radius_m,
            config.conductor_outer_radius_m,
            config.radial_cells_per_region,
        ),
        (
            config.conductor_outer_radius_m,
            config.outer_radius_m,
            config.radial_cells_per_region,
        ),
    ]);
    let axial = segmented_coordinates(&[
        (
            -config.axial_half_height_m,
            -config.conductor_axial_half_height_m,
            config.axial_cells_per_region,
        ),
        (
            -config.conductor_axial_half_height_m,
            config.conductor_axial_half_height_m,
            config.axial_cells_per_region,
        ),
        (
            config.conductor_axial_half_height_m,
            config.axial_half_height_m,
            config.axial_cells_per_region,
        ),
    ]);
    let angles: Vec<_> = (0..=config.circumferential_cells)
        .map(|index| config.sector_angle_rad() * index as f64 / config.circumferential_cells as f64)
        .collect();
    let nr = radial.len();
    let nt = angles.len();
    let node = |ir: usize, it: usize, iz: usize| (iz * nt + it) * nr + ir;
    let mut mesh = Mesh::new(3);
    for &z in &axial {
        for &theta in &angles {
            for &radius in &radial {
                mesh.add_node(Point::new_3d(radius * theta.cos(), radius * theta.sin(), z));
            }
        }
    }
    for iz in 0..axial.len() - 1 {
        for it in 0..angles.len() - 1 {
            for ir in 0..radial.len() - 1 {
                let r = 0.5 * (radial[ir] + radial[ir + 1]);
                let z = 0.5 * (axial[iz] + axial[iz + 1]);
                let marker = sector_cell_marker(config, r, z);
                let v000 = node(ir, it, iz);
                let v100 = node(ir + 1, it, iz);
                let v010 = node(ir, it + 1, iz);
                let v110 = node(ir + 1, it + 1, iz);
                let v001 = node(ir, it, iz + 1);
                let v101 = node(ir + 1, it, iz + 1);
                let v011 = node(ir, it + 1, iz + 1);
                let v111 = node(ir + 1, it + 1, iz + 1);
                for tetrahedron in [
                    [v000, v100, v110, v111],
                    [v000, v110, v010, v111],
                    [v000, v010, v011, v111],
                    [v000, v011, v001, v111],
                    [v000, v001, v101, v111],
                    [v000, v101, v100, v111],
                ] {
                    mesh.add_element_with_marker(
                        ElementType::Tetrahedron,
                        tetrahedron.to_vec(),
                        marker,
                    );
                }
            }
        }
    }
    add_sector_boundaries(&mut mesh, config);
    for (dimension, tag, name) in [
        (3, SECTOR_AIR, "sector magnetic air"),
        (3, SECTOR_CONDUCTOR, "sector conductor"),
        (2, SECTOR_MASTER_FACE, "cyclic master face"),
        (2, SECTOR_SLAVE_FACE, "cyclic slave face"),
        (2, SECTOR_INNER_RADIAL, "inner magnetic truncation"),
        (2, SECTOR_OUTER_RADIAL, "outer magnetic truncation"),
        (2, SECTOR_AXIAL_ENDS, "axial magnetic truncation"),
    ] {
        mesh.register_physical_group(dimension, tag, name)?;
    }
    Ok(mesh)
}

fn build_axisymmetric_reference_mesh(config: &Sector3dConfig) -> Mesh {
    let radial = segmented_coordinates(&[
        (
            config.inner_radius_m,
            config.conductor_inner_radius_m,
            config.radial_cells_per_region,
        ),
        (
            config.conductor_inner_radius_m,
            config.conductor_outer_radius_m,
            config.radial_cells_per_region,
        ),
        (
            config.conductor_outer_radius_m,
            config.outer_radius_m,
            config.radial_cells_per_region,
        ),
    ]);
    let axial = segmented_coordinates(&[
        (
            -config.axial_half_height_m,
            -config.conductor_axial_half_height_m,
            config.axial_cells_per_region,
        ),
        (
            -config.conductor_axial_half_height_m,
            config.conductor_axial_half_height_m,
            config.axial_cells_per_region,
        ),
        (
            config.conductor_axial_half_height_m,
            config.axial_half_height_m,
            config.axial_cells_per_region,
        ),
    ]);
    let nr = radial.len();
    let mut mesh = Mesh::new(2);
    for &z in &axial {
        for &r in &radial {
            mesh.add_node(Point::new_2d(r, z));
        }
    }
    for iz in 0..axial.len() - 1 {
        for ir in 0..radial.len() - 1 {
            let marker = sector_cell_marker(
                config,
                0.5 * (radial[ir] + radial[ir + 1]),
                0.5 * (axial[iz] + axial[iz + 1]),
            );
            let n00 = iz * nr + ir;
            let n10 = n00 + 1;
            let n01 = (iz + 1) * nr + ir;
            let n11 = n01 + 1;
            mesh.add_element_with_marker(ElementType::Triangle, vec![n00, n10, n11], marker);
            mesh.add_element_with_marker(ElementType::Triangle, vec![n00, n11, n01], marker);
        }
    }
    mesh
}

fn add_sector_boundaries(mesh: &mut Mesh, config: &Sector3dConfig) {
    type FaceOwner = (usize, usize, Vec<usize>);
    type FaceOwners = BTreeMap<Vec<usize>, Vec<FaceOwner>>;
    let mut owners: FaceOwners = BTreeMap::new();
    for (element_index, element) in mesh.elements.iter().enumerate() {
        for (local_face, face) in element.faces().into_iter().enumerate() {
            let mut key = face.clone();
            key.sort_unstable();
            owners
                .entry(key)
                .or_default()
                .push((element_index, local_face, face));
        }
    }
    let tolerance = 1.0e-10;
    for adjacent in owners.into_values().filter(|owners| owners.len() == 1) {
        let (element, local_face, nodes) = adjacent.into_iter().next().unwrap();
        let points: Vec<_> = nodes.iter().map(|node| &mesh.nodes[*node]).collect();
        let on_master = points.iter().all(|point| point.y.abs() < tolerance);
        let angle = config.sector_angle_rad();
        let on_slave = points.iter().all(|point| {
            let cross = -point.x * angle.sin() + point.y * angle.cos();
            cross.abs() < tolerance
        });
        let on_inner = points
            .iter()
            .all(|point| (point.x.hypot(point.y) - config.inner_radius_m).abs() < tolerance);
        let on_outer = points
            .iter()
            .all(|point| (point.x.hypot(point.y) - config.outer_radius_m).abs() < tolerance);
        let marker = if on_master {
            SECTOR_MASTER_FACE
        } else if on_slave {
            SECTOR_SLAVE_FACE
        } else if on_inner {
            SECTOR_INNER_RADIAL
        } else if on_outer {
            SECTOR_OUTER_RADIAL
        } else {
            SECTOR_AXIAL_ENDS
        };
        let boundary_type = if matches!(marker, SECTOR_MASTER_FACE | SECTOR_SLAVE_FACE) {
            BoundaryType::Interior
        } else {
            BoundaryType::Dirichlet
        };
        mesh.add_boundary(nodes, boundary_type, marker, element, local_face);
    }
}

fn segmented_coordinates(segments: &[(f64, f64, usize)]) -> Vec<f64> {
    let mut coordinates = Vec::new();
    for &(start, end, cells) in segments {
        if coordinates.is_empty() {
            coordinates.push(start);
        }
        for index in 1..=cells {
            coordinates.push(start + (end - start) * index as f64 / cells as f64);
        }
    }
    coordinates
}

fn sector_cell_marker(config: &Sector3dConfig, radius: f64, axial: f64) -> i32 {
    if radius > config.conductor_inner_radius_m
        && radius < config.conductor_outer_radius_m
        && axial.abs() < config.conductor_axial_half_height_m
    {
        SECTOR_CONDUCTOR
    } else {
        SECTOR_AIR
    }
}

fn azimuthal_source(
    marker: i32,
    point: [f64; 3],
    source_marker: i32,
    magnitude: f64,
    circumferential_order: i32,
) -> [Complex64; 3] {
    if marker != source_marker {
        return [Complex64::default(); 3];
    }
    let radius = point[0].hypot(point[1]);
    let theta = point[1].atan2(point[0]);
    let phase = Complex64::from_polar(1.0, circumferential_order as f64 * theta);
    [
        phase * (-magnitude * point[1] / radius),
        phase * (magnitude * point[0] / radius),
        Complex64::default(),
    ]
}

fn complex_is_finite(value: Complex64) -> bool {
    value.re.is_finite() && value.im.is_finite()
}

#[derive(Debug, thiserror::Error)]
pub enum Sector3dError {
    #[error("invalid 3D sector configuration: {0}")]
    InvalidConfiguration(&'static str),
    #[error("no active non-slave scalar-potential reference node exists")]
    MissingScalarReference,
    #[error("axisymmetric comparison is only defined for circumferential order m=0")]
    ComparisonRequiresM0,
    #[error("3D sector solve returned a non-finite field")]
    NonFiniteSolution,
    #[error(transparent)]
    PhysicalGroup(#[from] PhysicalGroupError),
    #[error(transparent)]
    Periodic(#[from] MpcError),
    #[error(transparent)]
    Aphi(#[from] AphiError),
    #[error(transparent)]
    Solver(#[from] SolverError),
    #[error(transparent)]
    Axisymmetric(
        #[from] sonium_fem::physics::electromagnetics::eddy_current::AxisymmetricEddyError,
    ),
    #[error(transparent)]
    Export(#[from] FieldExportError),
}

#[cfg(test)]
mod tests {
    use super::*;

    fn coarse_model() -> Sector3dModel {
        Sector3dModel::build(Sector3dConfig::default()).unwrap()
    }

    #[test]
    fn sector_mesh_has_conforming_marked_periodic_faces() {
        let model = coarse_model();
        assert!(model.mesh.cells_with_marker(SECTOR_AIR).next().is_some());
        assert!(
            model
                .mesh
                .cells_with_marker(SECTOR_CONDUCTOR)
                .next()
                .is_some()
        );
        assert!(!model.periodic_map.node_pairs.is_empty());
        assert!(!model.periodic_map.edge_pairs.is_empty());
        assert_eq!(
            model.periodic_map.node_pairs.len(),
            model
                .mesh
                .boundaries
                .iter()
                .filter(|face| face.marker == SECTOR_MASTER_FACE)
                .flat_map(|face| face.nodes.iter().copied())
                .collect::<BTreeSet<_>>()
                .len()
        );
    }

    #[test]
    fn cartesian_vector_constraints_apply_the_sector_rotation() {
        let model = coarse_model();
        let constraints = model.periodic_map.rotated_vector_constraints(3).unwrap();
        let mut values = vec![Complex64::default(); 3 * model.mesh.num_nodes()];
        for (node, point) in model.mesh.nodes.iter().enumerate() {
            values[3 * node] = Complex64::new(point.x, 0.0);
            values[3 * node + 1] = Complex64::new(point.y, 0.0);
        }
        let max_residual = constraints
            .iter()
            .map(|constraint| {
                let expected: Complex64 = constraint
                    .masters
                    .iter()
                    .map(|term| term.coefficient * values[term.dof])
                    .sum();
                (values[constraint.slave] - expected).norm()
            })
            .fold(0.0, f64::max);
        assert!(max_residual < 1.0e-12);
    }

    #[test]
    fn a_phi_sector_solution_is_periodic_gauged_and_passive() {
        let model = coarse_model();
        let solution = model.solve_frequency(1_000.0).unwrap();
        assert!(solution.periodic_residual_max < 1.0e-10);
        assert!(solution.gauge_residual_max < 1.0e-10);
        assert!(solution.charge_continuity_relative_residual < 1.0e-9);
        assert!(solution.full_revolution_eddy_loss_w >= 0.0);
        assert!(solution.full_revolution_magnetic_energy_j > 0.0);
        assert!(solution.magnetic_impedance_ohm.re >= -1.0e-10);
        assert!(solution.magnetic_impedance_ohm.im > 0.0);
        let power_scale = solution.full_revolution_eddy_loss_w.max(1.0e-20);
        assert!(
            (solution.magnetic_impedance_ohm.re * model.config.current_rms_a.powi(2)
                - solution.full_revolution_eddy_loss_w)
                .abs()
                / power_scale
                < 0.08
        );
        assert!(solution.fields.cells.iter().any(|cell| {
            cell.current_density_a_per_m2
                .iter()
                .any(|value| value.norm() > 0.0)
        }));
    }

    #[test]
    fn current_scaling_is_linear_in_fields_and_quadratic_in_loss() {
        let unit = coarse_model().solve_frequency(700.0).unwrap();
        let doubled_model = Sector3dModel::build(Sector3dConfig {
            current_rms_a: 2.0,
            ..Sector3dConfig::default()
        })
        .unwrap();
        let doubled = doubled_model.solve_frequency(700.0).unwrap();
        assert!(
            (doubled.full_revolution_eddy_loss_w / unit.full_revolution_eddy_loss_w - 4.0).abs()
                < 1.0e-8
        );
        assert!(
            (doubled.magnetic_impedance_ohm / unit.magnetic_impedance_ohm
                - Complex64::new(1.0, 0.0))
            .norm()
                < 1.0e-8
        );
    }

    #[test]
    fn nonzero_circumferential_mode_uses_complex_floquet_phase() {
        let model = Sector3dModel::build(Sector3dConfig {
            circumferential_order: 1,
            ..Sector3dConfig::default()
        })
        .unwrap();
        assert!(model.periodic_map.phase.im.abs() > 0.0);
        let solution = model.solve_frequency(600.0).unwrap();
        assert!(solution.periodic_residual_max < 1.0e-10);
        assert!(solution.full_revolution_eddy_loss_w >= 0.0);
    }

    #[test]
    fn m0_sector_tracks_independent_axisymmetric_a_theta_reference() {
        let model = coarse_model();
        let comparison = model.compare_m0_axisymmetric(500.0).unwrap();
        // Independent lowest-order H(curl) tetrahedra and P1 A-theta
        // triangles converge from different sides on this intentionally coarse
        // mesh; this gate catches normalization/sign regressions without
        // pretending the coarse discretizations are mesh-converged.
        assert!(comparison.impedance_relative_error < 0.30, "{comparison:?}");
        assert!(comparison.eddy_loss_relative_error < 0.30, "{comparison:?}");
    }

    #[test]
    fn m0_sector_reference_error_decreases_with_circumferential_refinement() {
        let coarse = coarse_model().compare_m0_axisymmetric(500.0).unwrap();
        let refined = Sector3dModel::build(Sector3dConfig {
            circumferential_cells: 4,
            ..Sector3dConfig::default()
        })
        .unwrap()
        .compare_m0_axisymmetric(500.0)
        .unwrap();
        assert!(
            refined.impedance_relative_error < coarse.impedance_relative_error,
            "impedance did not converge: coarse={coarse:?}, refined={refined:?}"
        );
        assert!(
            refined.eddy_loss_relative_error < coarse.eddy_loss_relative_error,
            "eddy loss did not converge: coarse={coarse:?}, refined={refined:?}"
        );
        assert!(refined.impedance_relative_error < 0.05, "{refined:?}");
        assert!(refined.eddy_loss_relative_error < 0.01, "{refined:?}");
    }

    #[test]
    fn sector_vtu_and_pvd_exports_include_all_electromagnetic_fields() {
        let model = coarse_model();
        let solution = model.solve_frequency(800.0).unwrap();
        let directory = tempfile::tempdir().unwrap();
        let snapshot = model
            .export_solution(&solution, directory.path().join("sector.vtu"))
            .unwrap();
        let vtu = std::fs::read_to_string(snapshot.vtu_path).unwrap();
        for name in [
            field_names::MAGNETIC_POTENTIAL,
            field_names::MAGNETIC_FLUX_DENSITY,
            "electric_scalar_potential",
            "electric_field",
            field_names::EDDY_CURRENT_DENSITY,
            field_names::EDDY_LOSS_DENSITY,
            field_names::REGION_MARKER,
        ] {
            assert!(vtu.contains(name), "missing {name}");
        }
        let pvd_path = directory.path().join("sector.pvd");
        model.export_sweep(&[400.0, 900.0], &pvd_path).unwrap();
        let pvd = std::fs::read_to_string(pvd_path).unwrap();
        assert!(pvd.contains("timestep=\"400\""));
        assert!(pvd.contains("timestep=\"900\""));
    }
}
