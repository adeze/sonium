//! Checked three-level mesh-convergence evidence for RoomSim responses.
//!
//! The comparison operates on already-produced [`SimulationOutput`] values.
//! It deliberately does not run a solver or infer a mesh size: callers must
//! provide coarse, medium, and fine responses from the same source/listener
//! setup.  Listener positions are not stored in `SimulationOutput`, so their
//! identities remain a caller-owned part of the topology contract.

use crate::SimulationOutput;
use crate::fem_runner::FemNodalField;
use num_complex::Complex64;
use sonium_xem::Point3D;

/// A solved response together with the physical mesh and listener metadata
/// needed to make a mesh-convergence comparison reproducible.
#[derive(Debug, Clone)]
pub struct MeshConvergenceLevel {
    /// Human-readable level label, for example `coarse`, `medium`, or `fine`.
    pub label: String,
    /// Number of mesh nodes used for this response.
    pub node_count: usize,
    /// Number of mesh elements used for this response.
    pub element_count: usize,
    /// Exact listener coordinates used to populate `output.pressures`.
    pub listener_positions: Vec<Point3D>,
    /// Fully solved response tensor for this level.
    pub output: SimulationOutput,
}

impl MeshConvergenceLevel {
    /// Build a convergence level directly from a retained CPU FEM field and
    /// its listener response, preventing caller-supplied mesh counts or
    /// listener coordinates from drifting away from the solved field.
    pub fn from_fem_nodal_field(
        label: impl Into<String>,
        field: &FemNodalField,
        output: SimulationOutput,
    ) -> Result<Self, String> {
        field.validate()?;
        output.require_solved("mesh-convergence FEM response")?;
        if field.frequencies != output.frequencies || field.source_names != output.source_names {
            return Err("mesh-convergence FEM field and response axes differ".into());
        }
        if output.pressures.len() != field.source_names.len()
            || output.pressures.iter().any(|source| {
                source.len() != field.listener_positions.len()
                    || source
                        .iter()
                        .any(|listener| listener.len() != field.frequencies.len())
            })
        {
            return Err("mesh-convergence FEM response listener axis differs from field".into());
        }
        Ok(Self {
            label: label.into(),
            node_count: field.mesh.nodes.len(),
            element_count: field.mesh.elements.len(),
            listener_positions: field.listener_positions.clone(),
            output,
        })
    }

    fn validate(&self, role: &str) -> Result<(), String> {
        if self.label.trim().is_empty() {
            return Err(format!("mesh-convergence {role} label must not be empty"));
        }
        if self.node_count == 0 || self.element_count == 0 {
            return Err(format!(
                "mesh-convergence {role} mesh counts must be positive"
            ));
        }
        if self.listener_positions.is_empty()
            || self.listener_positions.iter().any(|position| {
                !position.x.is_finite() || !position.y.is_finite() || !position.z.is_finite()
            })
        {
            return Err(format!(
                "mesh-convergence {role} listener positions must be finite and non-empty"
            ));
        }
        if self
            .output
            .pressures
            .iter()
            .any(|source| source.len() != self.listener_positions.len())
        {
            return Err(format!(
                "mesh-convergence {role} listener axis differs from metadata"
            ));
        }
        self.output
            .require_solved(&format!("mesh-convergence {role} response"))
    }
}

/// A validated coarse/medium/fine physical FEM mesh ladder.
///
/// The ladder owns the retained-field-derived mesh and listener metadata for
/// every level, so callers cannot accidentally compare response tensors from
/// one mesh against counts or receiver coordinates from another.
#[derive(Debug, Clone)]
pub struct PhysicalMeshConvergenceLadder {
    pub coarse: MeshConvergenceLevel,
    pub medium: MeshConvergenceLevel,
    pub fine: MeshConvergenceLevel,
}

impl PhysicalMeshConvergenceLadder {
    /// Build a ladder directly from retained FEM fields and their solved
    /// listener-response outputs.
    pub fn from_fem_nodal_fields(
        coarse_label: impl Into<String>,
        coarse_field: &FemNodalField,
        coarse_output: SimulationOutput,
        medium_label: impl Into<String>,
        medium_field: &FemNodalField,
        medium_output: SimulationOutput,
        fine_label: impl Into<String>,
        fine_field: &FemNodalField,
        fine_output: SimulationOutput,
    ) -> Result<Self, String> {
        Self::from_levels(
            MeshConvergenceLevel::from_fem_nodal_field(coarse_label, coarse_field, coarse_output)?,
            MeshConvergenceLevel::from_fem_nodal_field(medium_label, medium_field, medium_output)?,
            MeshConvergenceLevel::from_fem_nodal_field(fine_label, fine_field, fine_output)?,
        )
    }

    /// Wrap already-constructed levels after checking their physical
    /// refinement metadata and labels.
    pub fn from_levels(
        coarse: MeshConvergenceLevel,
        medium: MeshConvergenceLevel,
        fine: MeshConvergenceLevel,
    ) -> Result<Self, String> {
        let ladder = Self {
            coarse,
            medium,
            fine,
        };
        ladder.validate()?;
        Ok(ladder)
    }

    /// Validate the level-local and refinement metadata without comparing
    /// response values.
    pub fn validate(&self) -> Result<(), String> {
        self.coarse.validate("coarse")?;
        self.medium.validate("medium")?;
        self.fine.validate("fine")?;

        let labels = [&self.coarse.label, &self.medium.label, &self.fine.label];
        if labels.iter().any(|label| label.trim().is_empty()) {
            return Err("mesh-convergence ladder labels must not be empty".into());
        }
        if labels[0] == labels[1] || labels[1] == labels[2] || labels[0] == labels[2] {
            return Err("mesh-convergence ladder labels must be distinct".into());
        }
        if self.coarse.node_count >= self.medium.node_count
            || self.medium.node_count >= self.fine.node_count
        {
            return Err(format!(
                "mesh-convergence ladder node counts must increase refinement (coarse={}, medium={}, fine={})",
                self.coarse.node_count, self.medium.node_count, self.fine.node_count
            ));
        }
        if self.coarse.element_count >= self.medium.element_count
            || self.medium.element_count >= self.fine.element_count
        {
            return Err(format!(
                "mesh-convergence ladder element counts must increase refinement (coarse={}, medium={}, fine={})",
                self.coarse.element_count, self.medium.element_count, self.fine.element_count
            ));
        }
        Ok(())
    }

    /// Compare the retained physical levels using the existing response
    /// topology, listener-position, and complex-pressure checks.
    pub fn compare(
        &self,
        listener_position_tolerance_m: f64,
        frequency_tolerance_hz: f64,
        relative_floor_pa: f64,
    ) -> Result<MeshConvergenceReport, String> {
        self.validate()?;
        compare_mesh_convergence_levels(
            &self.coarse,
            &self.medium,
            &self.fine,
            listener_position_tolerance_m,
            frequency_tolerance_hz,
            relative_floor_pa,
        )
    }

    /// Compare and enforce the fine-grid convergence budgets.
    pub fn require_converged(
        &self,
        listener_position_tolerance_m: f64,
        frequency_tolerance_hz: f64,
        relative_floor_pa: f64,
        max_rms_relative_error: f64,
        max_max_relative_error: f64,
    ) -> Result<MeshConvergenceReport, String> {
        let report = self.compare(
            listener_position_tolerance_m,
            frequency_tolerance_hz,
            relative_floor_pa,
        )?;
        report.require_converged(max_rms_relative_error, max_max_relative_error)?;
        Ok(report)
    }
}

/// Compare three solved FEM response levels while checking physical mesh
/// refinement and exact listener-position metadata.
pub fn compare_mesh_convergence_levels(
    coarse: &MeshConvergenceLevel,
    medium: &MeshConvergenceLevel,
    fine: &MeshConvergenceLevel,
    listener_position_tolerance_m: f64,
    frequency_tolerance_hz: f64,
    relative_floor_pa: f64,
) -> Result<MeshConvergenceReport, String> {
    if !listener_position_tolerance_m.is_finite() || listener_position_tolerance_m < 0.0 {
        return Err(
            "mesh-convergence listener position tolerance must be finite non-negative".into(),
        );
    }
    coarse.validate("coarse")?;
    medium.validate("medium")?;
    fine.validate("fine")?;
    if coarse.node_count >= medium.node_count || medium.node_count >= fine.node_count {
        return Err("mesh-convergence node counts must increase with refinement".into());
    }
    if coarse.element_count >= medium.element_count || medium.element_count >= fine.element_count {
        return Err("mesh-convergence element counts must increase with refinement".into());
    }
    validate_listener_positions(
        &coarse.listener_positions,
        &medium.listener_positions,
        listener_position_tolerance_m,
        "coarse/medium",
    )?;
    validate_listener_positions(
        &medium.listener_positions,
        &fine.listener_positions,
        listener_position_tolerance_m,
        "medium/fine",
    )?;
    compare_mesh_convergence(
        &coarse.output,
        &medium.output,
        &fine.output,
        frequency_tolerance_hz,
        relative_floor_pa,
    )
}

fn validate_listener_positions(
    coarser: &[Point3D],
    finer: &[Point3D],
    tolerance_m: f64,
    label: &str,
) -> Result<(), String> {
    if coarser.len() != finer.len() {
        return Err(format!(
            "mesh-convergence {label} listener-position count mismatch"
        ));
    }
    for (index, (coarser, finer)) in coarser.iter().zip(finer).enumerate() {
        let distance = ((coarser.x - finer.x).powi(2)
            + (coarser.y - finer.y).powi(2)
            + (coarser.z - finer.z).powi(2))
        .sqrt();
        if !distance.is_finite() || distance > tolerance_m {
            return Err(format!(
                "mesh-convergence {label} listener position {index} differs by {distance} m"
            ));
        }
    }
    Ok(())
}

/// Relative complex-pressure errors between adjacent response resolutions.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MeshConvergenceReport {
    /// Number of source/listener/frequency samples compared at each level.
    pub sample_count: usize,
    /// RMS relative complex-pressure error from coarse to medium.
    pub coarse_to_medium_rms_relative_error: f64,
    /// Maximum relative complex-pressure error from coarse to medium.
    pub coarse_to_medium_max_relative_error: f64,
    /// RMS relative complex-pressure error from medium to fine.
    pub medium_to_fine_rms_relative_error: f64,
    /// Maximum relative complex-pressure error from medium to fine.
    pub medium_to_fine_max_relative_error: f64,
}

impl MeshConvergenceReport {
    /// Whether both aggregate error measures decrease with refinement.
    pub fn is_monotonic(&self) -> bool {
        self.medium_to_fine_rms_relative_error <= self.coarse_to_medium_rms_relative_error
            && self.medium_to_fine_max_relative_error <= self.coarse_to_medium_max_relative_error
    }

    /// Require monotonic refinement and a bounded medium-to-fine error.
    ///
    /// The fine-grid comparison is the acceptance result; the coarse-to-medium
    /// comparison is retained to show that the final refinement improved the
    /// response rather than merely passing a one-step tolerance.
    pub fn require_converged(
        &self,
        max_rms_relative_error: f64,
        max_max_relative_error: f64,
    ) -> Result<(), String> {
        if !max_rms_relative_error.is_finite()
            || max_rms_relative_error < 0.0
            || !max_max_relative_error.is_finite()
            || max_max_relative_error < 0.0
        {
            return Err("mesh-convergence error budgets must be finite non-negative values".into());
        }
        if self.sample_count == 0 {
            return Err("mesh-convergence report requires at least one sample".into());
        }
        let metrics = [
            self.coarse_to_medium_rms_relative_error,
            self.coarse_to_medium_max_relative_error,
            self.medium_to_fine_rms_relative_error,
            self.medium_to_fine_max_relative_error,
        ];
        if metrics
            .iter()
            .any(|metric| !metric.is_finite() || *metric < 0.0)
        {
            return Err("mesh-convergence report contains invalid error metrics".into());
        }
        if !self.is_monotonic() {
            return Err(format!(
                "mesh-convergence error is not monotonic: coarse/medium RMS={} max={}, medium/fine RMS={} max={}",
                self.coarse_to_medium_rms_relative_error,
                self.coarse_to_medium_max_relative_error,
                self.medium_to_fine_rms_relative_error,
                self.medium_to_fine_max_relative_error,
            ));
        }
        if self.medium_to_fine_rms_relative_error > max_rms_relative_error
            || self.medium_to_fine_max_relative_error > max_max_relative_error
        {
            return Err(format!(
                "mesh-convergence fine-grid error exceeds budget: RMS={} (limit {}), max={} (limit {})",
                self.medium_to_fine_rms_relative_error,
                max_rms_relative_error,
                self.medium_to_fine_max_relative_error,
                max_max_relative_error,
            ));
        }
        Ok(())
    }
}

/// Compare coarse, medium, and fine solved response tensors.
///
/// Each adjacent error is normalized by the finer pressure magnitude, with
/// `relative_floor_pa` preventing division by zero at pressure nulls.  A zero
/// difference remains exactly zero even when both pressures are zero.
pub fn compare_mesh_convergence(
    coarse: &SimulationOutput,
    medium: &SimulationOutput,
    fine: &SimulationOutput,
    frequency_tolerance_hz: f64,
    relative_floor_pa: f64,
) -> Result<MeshConvergenceReport, String> {
    if !frequency_tolerance_hz.is_finite() || frequency_tolerance_hz < 0.0 {
        return Err("mesh-convergence frequency tolerance must be finite non-negative".into());
    }
    if !relative_floor_pa.is_finite() || relative_floor_pa <= 0.0 {
        return Err("mesh-convergence relative pressure floor must be finite positive".into());
    }

    coarse.require_solved("mesh-convergence coarse response")?;
    medium.require_solved("mesh-convergence medium response")?;
    fine.require_solved("mesh-convergence fine response")?;
    validate_topology(coarse, medium, frequency_tolerance_hz, "coarse/medium")?;
    validate_topology(medium, fine, frequency_tolerance_hz, "medium/fine")?;

    let coarse_to_medium = compare_pair(coarse, medium, relative_floor_pa, "coarse/medium")?;
    let medium_to_fine = compare_pair(medium, fine, relative_floor_pa, "medium/fine")?;
    Ok(MeshConvergenceReport {
        sample_count: coarse_to_medium.0,
        coarse_to_medium_rms_relative_error: coarse_to_medium.1,
        coarse_to_medium_max_relative_error: coarse_to_medium.2,
        medium_to_fine_rms_relative_error: medium_to_fine.1,
        medium_to_fine_max_relative_error: medium_to_fine.2,
    })
}

fn validate_topology(
    coarser: &SimulationOutput,
    finer: &SimulationOutput,
    frequency_tolerance_hz: f64,
    label: &str,
) -> Result<(), String> {
    if coarser.source_names != finer.source_names {
        return Err(format!("mesh-convergence {label} source names differ"));
    }
    if coarser.frequencies.len() != finer.frequencies.len() {
        return Err(format!(
            "mesh-convergence {label} frequency grids differ in length"
        ));
    }
    for (index, (coarser_frequency, finer_frequency)) in coarser
        .frequencies
        .iter()
        .zip(&finer.frequencies)
        .enumerate()
    {
        if (coarser_frequency - finer_frequency).abs() > frequency_tolerance_hz {
            return Err(format!(
                "mesh-convergence {label} frequency mismatch at index {index}: coarse={coarser_frequency}, fine={finer_frequency}"
            ));
        }
    }
    for (source_index, (coarser_source, finer_source)) in
        coarser.pressures.iter().zip(&finer.pressures).enumerate()
    {
        if coarser_source.len() != finer_source.len() {
            return Err(format!(
                "mesh-convergence {label} listener axes differ for source {source_index}"
            ));
        }
    }
    Ok(())
}

fn compare_pair(
    coarser: &SimulationOutput,
    finer: &SimulationOutput,
    relative_floor_pa: f64,
    label: &str,
) -> Result<(usize, f64, f64), String> {
    let mut sample_count = 0usize;
    let mut sum_squared = 0.0f64;
    let mut maximum = 0.0f64;
    for (source_index, (coarser_source, finer_source)) in
        coarser.pressures.iter().zip(&finer.pressures).enumerate()
    {
        for (listener_index, (coarser_response, finer_response)) in
            coarser_source.iter().zip(finer_source).enumerate()
        {
            for (frequency_index, (&coarser_pressure, &finer_pressure)) in
                coarser_response.iter().zip(finer_response).enumerate()
            {
                let error =
                    relative_complex_error(coarser_pressure, finer_pressure, relative_floor_pa);
                if !error.is_finite() {
                    return Err(format!(
                        "mesh-convergence {label} relative error is non-finite at source {source_index}, listener {listener_index}, frequency {frequency_index}"
                    ));
                }
                sample_count = sample_count
                    .checked_add(1)
                    .ok_or_else(|| "mesh-convergence sample count overflowed".to_string())?;
                sum_squared += error * error;
                if !sum_squared.is_finite() {
                    return Err(format!(
                        "mesh-convergence {label} RMS accumulation is non-finite"
                    ));
                }
                maximum = maximum.max(error);
            }
        }
    }
    if sample_count == 0 {
        return Err(format!("mesh-convergence {label} contains no samples"));
    }
    let rms = (sum_squared / sample_count as f64).sqrt();
    if !rms.is_finite() || !maximum.is_finite() {
        return Err(format!("mesh-convergence {label} report is non-finite"));
    }
    Ok((sample_count, rms, maximum))
}

fn relative_complex_error(coarser: Complex64, finer: Complex64, relative_floor_pa: f64) -> f64 {
    let difference = (coarser - finer).norm();
    if difference == 0.0 {
        return 0.0;
    }
    difference / finer.norm().max(relative_floor_pa)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::FrequencyProvenance;
    use sonium_fem::mesh::{ElementType, Mesh, Point};

    fn response(values: &[f64], provenance: FrequencyProvenance) -> SimulationOutput {
        SimulationOutput::with_uniform_provenance(
            vec![100.0, 200.0, 400.0],
            vec![vec![
                values
                    .iter()
                    .map(|value| Complex64::new(*value, *value * 0.1))
                    .collect(),
            ]],
            vec!["source".into()],
            provenance,
        )
    }

    #[test]
    fn converged_report_requires_monotonic_fine_grid_error_budget() {
        let report = compare_mesh_convergence(
            &response(&[10.0, 20.0, 30.0], FrequencyProvenance::Solved),
            &response(&[9.0, 18.0, 27.0], FrequencyProvenance::Solved),
            &response(&[8.5, 17.0, 25.5], FrequencyProvenance::Solved),
            0.0,
            1.0e-12,
        )
        .unwrap();
        assert_eq!(report.sample_count, 3);
        assert!(report.is_monotonic());
        report.require_converged(0.07, 0.07).unwrap();
    }

    #[test]
    fn non_monotonic_refinement_is_rejected() {
        let report = compare_mesh_convergence(
            &response(&[10.0, 20.0, 30.0], FrequencyProvenance::Solved),
            &response(&[9.9, 19.8, 29.7], FrequencyProvenance::Solved),
            &response(&[8.0, 16.0, 24.0], FrequencyProvenance::Solved),
            0.0,
            1.0e-12,
        )
        .unwrap();
        let error = report.require_converged(1.0, 1.0).unwrap_err();
        assert!(error.contains("not monotonic"));
    }

    #[test]
    fn topology_mismatch_is_rejected() {
        let coarse = response(&[10.0, 20.0, 30.0], FrequencyProvenance::Solved);
        let mut medium = response(&[9.0, 18.0, 27.0], FrequencyProvenance::Solved);
        medium.source_names[0] = "other-source".into();
        let fine = response(&[8.5, 17.0, 25.5], FrequencyProvenance::Solved);
        let error = compare_mesh_convergence(&coarse, &medium, &fine, 0.0, 1.0e-12).unwrap_err();
        assert!(error.contains("source names differ"));
    }

    #[test]
    fn listener_axis_mismatch_is_rejected() {
        let coarse = response(&[10.0, 20.0, 30.0], FrequencyProvenance::Solved);
        let mut medium = response(&[9.0, 18.0, 27.0], FrequencyProvenance::Solved);
        let extra_listener = medium.pressures[0][0].clone();
        medium.pressures[0].push(extra_listener);
        let fine = response(&[8.5, 17.0, 25.5], FrequencyProvenance::Solved);
        let error = compare_mesh_convergence(&coarse, &medium, &fine, 0.0, 1.0e-12).unwrap_err();
        assert!(error.contains("listener axes differ"));
    }

    #[test]
    fn non_solved_or_non_finite_responses_are_rejected() {
        let coarse = response(&[10.0, 20.0, 30.0], FrequencyProvenance::Solved);
        let medium = response(&[9.0, 18.0, 27.0], FrequencyProvenance::ReducedOrder);
        let fine = response(&[8.5, 17.0, 25.5], FrequencyProvenance::Solved);
        let error = compare_mesh_convergence(&coarse, &medium, &fine, 0.0, 1.0e-12).unwrap_err();
        assert!(error.contains("requires solved samples"));

        let mut non_finite = response(&[9.0, 18.0, 27.0], FrequencyProvenance::Solved);
        non_finite.pressures[0][0][1] = Complex64::new(f64::NAN, 0.0);
        let error =
            compare_mesh_convergence(&coarse, &non_finite, &fine, 0.0, 1.0e-12).unwrap_err();
        assert!(error.contains("non-finite pressure"));
    }

    #[test]
    fn invalid_tolerances_and_budgets_are_rejected() {
        let coarse = response(&[10.0, 20.0, 30.0], FrequencyProvenance::Solved);
        let medium = response(&[9.0, 18.0, 27.0], FrequencyProvenance::Solved);
        let fine = response(&[8.5, 17.0, 25.5], FrequencyProvenance::Solved);
        assert!(
            compare_mesh_convergence(&coarse, &medium, &fine, -1.0, 1.0e-12)
                .unwrap_err()
                .contains("frequency tolerance")
        );
        assert!(
            compare_mesh_convergence(&coarse, &medium, &fine, 0.0, 0.0)
                .unwrap_err()
                .contains("pressure floor")
        );
        let report = compare_mesh_convergence(&coarse, &medium, &fine, 0.0, 1.0e-12).unwrap();
        assert!(
            report
                .require_converged(f64::NAN, 0.1)
                .unwrap_err()
                .contains("budgets")
        );
    }

    fn level(
        label: &str,
        node_count: usize,
        element_count: usize,
        output: SimulationOutput,
    ) -> MeshConvergenceLevel {
        MeshConvergenceLevel {
            label: label.into(),
            node_count,
            element_count,
            listener_positions: vec![Point3D::new(0.5, 0.5, 0.5)],
            output,
        }
    }

    #[test]
    fn physical_mesh_ladder_wraps_levels_and_enforces_budgets() {
        let ladder = PhysicalMeshConvergenceLadder::from_levels(
            level(
                "coarse",
                10,
                4,
                response(&[10.0, 20.0, 30.0], FrequencyProvenance::Solved),
            ),
            level(
                "medium",
                20,
                8,
                response(&[9.0, 18.0, 27.0], FrequencyProvenance::Solved),
            ),
            level(
                "fine",
                40,
                16,
                response(&[8.5, 17.0, 25.5], FrequencyProvenance::Solved),
            ),
        )
        .unwrap();

        assert_eq!(ladder.coarse.node_count, 10);
        assert_eq!(ladder.medium.element_count, 8);
        assert_eq!(
            ladder.coarse.listener_positions,
            ladder.fine.listener_positions
        );
        let report = ladder
            .require_converged(1.0e-12, 0.0, 1.0e-12, 0.07, 0.07)
            .unwrap();
        assert!(report.is_monotonic());

        let mut duplicate_label = ladder.medium.clone();
        duplicate_label.label = "coarse".into();
        let error = PhysicalMeshConvergenceLadder::from_levels(
            ladder.coarse.clone(),
            duplicate_label,
            ladder.fine.clone(),
        )
        .unwrap_err();
        assert!(error.contains("labels must be distinct"));
    }

    #[test]
    fn physical_mesh_ladder_builds_from_retained_fields() {
        fn field_with_mesh(node_count: usize, element_count: usize) -> FemNodalField {
            let mut mesh = Mesh::new(3);
            for point in [
                Point::new_3d(0.0, 0.0, 0.0),
                Point::new_3d(1.0, 0.0, 0.0),
                Point::new_3d(0.0, 1.0, 0.0),
                Point::new_3d(0.0, 0.0, 1.0),
                Point::new_3d(1.0, 1.0, 0.0),
                Point::new_3d(1.0, 0.0, 1.0),
            ]
            .into_iter()
            .take(node_count)
            {
                mesh.add_node(point);
            }
            for _ in 0..element_count {
                mesh.add_element(ElementType::Tetrahedron, vec![0, 1, 2, 3]);
            }
            FemNodalField {
                mesh,
                frequencies: vec![100.0, 200.0, 400.0],
                source_names: vec!["source".into()],
                listener_positions: vec![Point3D::new(0.25, 0.25, 0.25)],
                pressures: vec![vec![
                    vec![Complex64::new(1.0, 0.0); node_count],
                    vec![Complex64::new(0.9, 0.0); node_count],
                    vec![Complex64::new(0.8, 0.0); node_count],
                ]],
            }
        }

        let coarse_field = field_with_mesh(4, 1);
        let medium_field = field_with_mesh(5, 2);
        let fine_field = field_with_mesh(6, 3);
        let ladder = PhysicalMeshConvergenceLadder::from_fem_nodal_fields(
            "coarse",
            &coarse_field,
            response(&[10.0, 20.0, 30.0], FrequencyProvenance::Solved),
            "medium",
            &medium_field,
            response(&[9.0, 18.0, 27.0], FrequencyProvenance::Solved),
            "fine",
            &fine_field,
            response(&[8.5, 17.0, 25.5], FrequencyProvenance::Solved),
        )
        .unwrap();
        assert_eq!(ladder.coarse.node_count, 4);
        assert_eq!(ladder.medium.element_count, 2);
        assert_eq!(
            ladder.fine.listener_positions,
            fine_field.listener_positions
        );
    }

    #[test]
    fn physical_mesh_levels_require_refinement_and_listener_identity() {
        let coarse = level(
            "coarse",
            10,
            4,
            response(&[10.0, 20.0, 30.0], FrequencyProvenance::Solved),
        );
        let medium = level(
            "medium",
            20,
            8,
            response(&[9.0, 18.0, 27.0], FrequencyProvenance::Solved),
        );
        let fine = level(
            "fine",
            40,
            16,
            response(&[8.5, 17.0, 25.5], FrequencyProvenance::Solved),
        );
        let report =
            compare_mesh_convergence_levels(&coarse, &medium, &fine, 1.0e-12, 0.0, 1.0e-12)
                .unwrap();
        assert!(report.is_monotonic());

        let mut shifted_medium = medium.clone();
        shifted_medium.listener_positions[0].x += 0.01;
        let error =
            compare_mesh_convergence_levels(&coarse, &shifted_medium, &fine, 1.0e-4, 0.0, 1.0e-12)
                .unwrap_err();
        assert!(error.contains("listener position"));
    }

    #[test]
    fn fem_field_constructor_binds_mesh_and_listener_metadata_to_response() {
        let mut mesh = Mesh::new(3);
        for point in [
            Point::new_3d(0.0, 0.0, 0.0),
            Point::new_3d(1.0, 0.0, 0.0),
            Point::new_3d(0.0, 1.0, 0.0),
            Point::new_3d(0.0, 0.0, 1.0),
        ] {
            mesh.add_node(point);
        }
        mesh.add_element(ElementType::Tetrahedron, vec![0, 1, 2, 3]);
        let field = FemNodalField {
            mesh,
            frequencies: vec![100.0, 200.0, 300.0],
            source_names: vec!["source".into()],
            listener_positions: vec![Point3D::new(0.5, 0.5, 0.5)],
            pressures: vec![vec![vec![Complex64::new(1.0, 0.0); 4]; 3]],
        };
        let mut field_response = response(&[10.0, 20.0, 30.0], FrequencyProvenance::Solved);
        field_response.frequencies[2] = 300.0;
        let level =
            MeshConvergenceLevel::from_fem_nodal_field("coarse", &field, field_response).unwrap();
        assert_eq!(level.node_count, 4);
        assert_eq!(level.element_count, 1);
        assert_eq!(level.listener_positions, field.listener_positions);

        let mut mismatched = response(&[10.0, 20.0, 30.0], FrequencyProvenance::Solved);
        mismatched.source_names[0] = "other".into();
        let error =
            MeshConvergenceLevel::from_fem_nodal_field("coarse", &field, mismatched).unwrap_err();
        assert!(error.contains("axes differ"));
    }

    #[test]
    fn physical_mesh_levels_reject_non_monotonic_counts_and_bad_positions() {
        let coarse = level(
            "coarse",
            20,
            8,
            response(&[10.0, 20.0, 30.0], FrequencyProvenance::Solved),
        );
        let medium = level(
            "medium",
            10,
            4,
            response(&[9.0, 18.0, 27.0], FrequencyProvenance::Solved),
        );
        let fine = level(
            "fine",
            40,
            16,
            response(&[8.5, 17.0, 25.5], FrequencyProvenance::Solved),
        );
        let error = compare_mesh_convergence_levels(&coarse, &medium, &fine, 0.0, 0.0, 1.0e-12)
            .unwrap_err();
        assert!(error.contains("node counts"));

        let mut invalid = fine;
        invalid.listener_positions[0].z = f64::NAN;
        let valid_medium = level(
            "medium",
            30,
            12,
            response(&[9.0, 18.0, 27.0], FrequencyProvenance::Solved),
        );
        let error =
            compare_mesh_convergence_levels(&coarse, &valid_medium, &invalid, 0.0, 0.0, 1.0e-12)
                .unwrap_err();
        assert!(error.contains("listener positions"));
    }
}
