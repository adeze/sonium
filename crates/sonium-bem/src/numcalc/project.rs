//! Load a Mesh2HRTF project directory into a runnable BEM problem.

use std::path::Path;

use num_complex::Complex64;

use crate::core::io::nc_format::{
    BoundarySpec, NcInputConfig, NcParseError, PlaneWaveSource, PointSource as NcPointSource,
    load_nc_elements, load_nc_nodes, parse_nc_input,
};
use crate::core::mesh::element::{
    compute_element_area, compute_element_center, compute_element_normal,
};
use crate::core::types::{
    BoundaryCondition, Element, ElementProperty, PhysicsParams, PhysicsParamsError,
};

/// Errors raised while loading a Mesh2HRTF project.
#[derive(Debug, thiserror::Error)]
pub enum ProjectError {
    /// Underlying NC.inp parse failure.
    #[error(transparent)]
    Parse(#[from] NcParseError),
    /// I/O error reading a file.
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    /// Boundary spec referenced an element index outside the mesh.
    #[error("boundary spec covers element {0} but mesh only has {1} elements")]
    BoundaryOutOfRange(usize, usize),
    /// Unrecognised boundary condition keyword.
    #[error("unsupported boundary condition keyword: {0}")]
    UnsupportedBc(String),
}

/// A fully-resolved Mesh2HRTF project ready to drive the BEM solver.
pub struct NumCalcProject {
    /// Original parsed NC.inp.
    pub config: NcInputConfig,
    /// Combined node coordinates from all object + evaluation grid files.
    pub nodes: Vec<[f64; 3]>,
    /// All elements (surface + evaluation), with BCs applied.
    pub elements: Vec<Element>,
    /// Indices into `elements` that are part of an evaluation grid.
    pub evaluation_indices: Vec<usize>,
    /// Frequencies (Hz), in solve order.
    pub frequencies: Vec<f64>,
    /// Plane-wave incident sources (may be empty).
    pub plane_waves: Vec<PlaneWaveSource>,
    /// Point incident sources (may be empty).
    pub point_sources: Vec<NcPointSource>,
    /// Speed of sound (m/s).
    pub speed_of_sound: f64,
    /// Air density (kg/m³).
    pub density: f64,
}

impl NumCalcProject {
    /// Build validated physical parameters for one frequency.
    pub fn try_physics_at(&self, frequency: f64) -> Result<PhysicsParams, PhysicsParamsError> {
        PhysicsParams::try_new(frequency, self.speed_of_sound, self.density, false)
    }

    /// Build `PhysicsParams` for one frequency.
    pub fn physics_at(&self, frequency: f64) -> PhysicsParams {
        PhysicsParams::new(frequency, self.speed_of_sound, self.density, false)
    }

    /// Iterator over (centre, normal) of each evaluation element.
    pub fn evaluation_centers(&self) -> Vec<[f64; 3]> {
        self.evaluation_indices
            .iter()
            .map(|&idx| self.elements[idx].center)
            .collect()
    }
}

/// Load a project directory: parse `NC.inp` (or the supplied path) and all
/// node / element files referenced from it.
pub fn load_project<P: AsRef<Path>>(nc_inp_path: P) -> Result<NumCalcProject, ProjectError> {
    let nc_path = nc_inp_path.as_ref();
    let config = parse_nc_input(nc_path)?;

    let (nodes, mut elements, evaluation_indices) = assemble_mesh(&config)?;
    apply_boundary_specs(&mut elements, &config.boundary_conditions)?;

    // Recompute geometry (centre, area, normal) now that BCs are in place.
    for elem in elements.iter_mut() {
        elem.center = compute_element_center(&nodes, &elem.connectivity);
        elem.area = compute_element_area(&nodes, &elem.connectivity, elem.element_type);
        elem.normal = compute_element_normal(&nodes, &elem.connectivity, elem.element_type);
    }

    let frequencies = expand_frequency_curve(&config);

    Ok(NumCalcProject {
        speed_of_sound: config.main_params_iv.speed_of_sound,
        density: config.main_params_iv.density,
        nodes,
        elements,
        evaluation_indices,
        frequencies,
        plane_waves: config.plane_waves.clone(),
        point_sources: config.point_sources.clone(),
        config,
    })
}

/// (nodes, elements, indices of evaluation-grid elements within `elements`).
type AssembledMesh = (Vec<[f64; 3]>, Vec<Element>, Vec<usize>);

/// Combine all object meshes + evaluation grid meshes into one set of nodes
/// and elements, returning the indices of evaluation elements in `elements`.
///
/// The NumCalc convention is that the first `num_object_files` entries in
/// the `NODES` / `ELEMENTS` lists are object meshes and the remaining
/// `num_eval_files` entries are evaluation grids. We tag the latter with
/// `ElementProperty::Evaluation` so the assembly path skips them and the
/// post-processor evaluates the field at their centroids.
fn assemble_mesh(config: &NcInputConfig) -> Result<AssembledMesh, ProjectError> {
    let mut all_nodes: Vec<[f64; 3]> = Vec::new();
    let mut all_elements: Vec<Element> = Vec::new();
    let mut eval_indices: Vec<usize> = Vec::new();

    let n_files = config.node_files.len().min(config.element_files.len());
    let n_object = (config.main_params_i.num_object_files as usize).min(n_files);

    // Pre-load per-file node counts so we don't re-read the same files when
    // computing element offsets.
    let mut chunk_node_counts: Vec<usize> = Vec::with_capacity(n_files);
    for nf in config.node_files.iter().take(n_files) {
        let chunk = load_nc_nodes(nf)?;
        chunk_node_counts.push(chunk.len());
        all_nodes.extend_from_slice(&chunk);
    }

    let mut node_offset = 0usize;
    for (i, ef) in config.element_files.iter().take(n_files).enumerate() {
        let property = if i < n_object {
            ElementProperty::Surface
        } else {
            ElementProperty::Evaluation
        };
        let chunk = load_nc_elements(ef, property)?;
        let appended_start = all_elements.len();
        offset_and_extend(&mut all_elements, chunk, node_offset);
        if matches!(property, ElementProperty::Evaluation) {
            for idx in appended_start..all_elements.len() {
                eval_indices.push(idx);
            }
        }
        node_offset += chunk_node_counts[i];
    }

    Ok((all_nodes, all_elements, eval_indices))
}

fn offset_and_extend(dst: &mut Vec<Element>, mut src: Vec<Element>, offset: usize) {
    if offset != 0 {
        for elem in src.iter_mut() {
            for n in elem.connectivity.iter_mut() {
                *n += offset;
            }
        }
    }
    // DOF addresses get rebuilt by `BemSolver::prepare_elements` callers, but
    // since we drive the lower-level frequency_sweep API directly, assign a
    // monotonically increasing address now.
    let mut next_dof = dst
        .iter()
        .flat_map(|e| e.dof_addresses.iter().copied())
        .max()
        .map(|m| m + 1)
        .unwrap_or(0);
    for mut elem in src {
        let n_dof = elem.dof_addresses.len().max(1);
        elem.dof_addresses = (next_dof..next_dof + n_dof).collect();
        next_dof += n_dof;
        dst.push(elem);
    }
}

/// Apply each `BoundarySpec` to the corresponding element range.
///
/// `ELEM a TO b VELO re ci im ci2` becomes a `BoundaryCondition::Velocity`
/// with the prescribed complex value on every element with `a <= idx <= b`.
fn apply_boundary_specs(
    elements: &mut [Element],
    specs: &[BoundarySpec],
) -> Result<(), ProjectError> {
    for spec in specs {
        if spec.elem_start > spec.elem_end {
            return Err(ProjectError::BoundaryOutOfRange(
                spec.elem_start,
                spec.elem_end,
            ));
        }
        if spec.elem_end >= elements.len() {
            return Err(ProjectError::BoundaryOutOfRange(
                spec.elem_end,
                elements.len(),
            ));
        }
        let value = Complex64::new(spec.value_re, spec.value_im);
        let bc = match spec.bc_type.as_str() {
            "VELO" => BoundaryCondition::Velocity(vec![value]),
            "PRES" => BoundaryCondition::Pressure(vec![value]),
            "ADMI" => BoundaryCondition::VelocityWithAdmittance {
                velocity: vec![Complex64::new(0.0, 0.0)],
                admittance: value,
            },
            other => return Err(ProjectError::UnsupportedBc(other.to_string())),
        };
        for elem in &mut elements[spec.elem_start..=spec.elem_end] {
            elem.boundary_condition = bc.clone();
        }
    }
    Ok(())
}

/// Expand the NC.inp frequency curve into the explicit frequency list.
///
/// Mesh2HRTF stores the curve as `(time, frequency, phase)` pairs sampled
/// from the GUI's frequency stepper. The frequencies actually solved are
/// every entry whose frequency value is strictly positive.
fn expand_frequency_curve(config: &NcInputConfig) -> Vec<f64> {
    // `is_finite() && f > 0.0` filters out NaN and ±inf so the subsequent
    // sort is safe (NaN has no total order and would crash `partial_cmp`).
    let mut freqs: Vec<f64> = config
        .frequency_curve
        .iter()
        .map(|(_, f, _)| *f)
        .filter(|f| f.is_finite() && *f > 0.0)
        .collect();
    freqs.sort_by(|a, b| a.partial_cmp(b).expect("NaNs filtered above"));
    freqs.dedup();
    freqs
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::io::nc_format::parse_nc_input_string;
    use std::path::PathBuf;

    const SAMPLE_NC: &str = include_str!("../../tests/data/sample_project/NC.inp");

    #[test]
    fn boundary_specs_are_applied_per_range() {
        let mut elements = vec![sample_element(); 6];
        let specs = vec![BoundarySpec {
            elem_start: 1,
            elem_end: 3,
            bc_type: "VELO".to_string(),
            value_re: 1.0,
            curve_re: -1,
            value_im: 0.0,
            curve_im: -1,
        }];
        apply_boundary_specs(&mut elements, &specs).unwrap();
        for (i, e) in elements.iter().enumerate() {
            match (i, &e.boundary_condition) {
                (1..=3, BoundaryCondition::Velocity(v)) => {
                    assert_eq!(v[0], Complex64::new(1.0, 0.0))
                }
                (_, BoundaryCondition::Velocity(v)) => assert_eq!(v[0], Complex64::new(0.0, 0.0)),
                _ => panic!("unexpected BC"),
            }
        }
    }

    #[test]
    fn frequency_curve_is_sorted_unique_positive() {
        let cfg = parse_nc_input_string(SAMPLE_NC, PathBuf::from(".")).unwrap();
        let freqs = expand_frequency_curve(&cfg);
        assert!(freqs.iter().all(|f| *f > 0.0));
        assert!(freqs.windows(2).all(|w| w[0] <= w[1]));
    }

    #[test]
    fn boundary_specs_reject_inverted_range() {
        let mut elements = vec![sample_element(); 4];
        let specs = vec![BoundarySpec {
            elem_start: 3,
            elem_end: 1,
            bc_type: "VELO".to_string(),
            value_re: 1.0,
            curve_re: -1,
            value_im: 0.0,
            curve_im: -1,
        }];
        let err = apply_boundary_specs(&mut elements, &specs).unwrap_err();
        assert!(matches!(err, ProjectError::BoundaryOutOfRange(3, 1)));
    }

    #[test]
    fn boundary_specs_reject_out_of_range() {
        let mut elements = vec![sample_element(); 4];
        let specs = vec![BoundarySpec {
            elem_start: 0,
            elem_end: 9,
            bc_type: "VELO".to_string(),
            value_re: 1.0,
            curve_re: -1,
            value_im: 0.0,
            curve_im: -1,
        }];
        assert!(apply_boundary_specs(&mut elements, &specs).is_err());
    }

    #[test]
    fn frequency_curve_drops_nan_and_negative() {
        let mut cfg = parse_nc_input_string(SAMPLE_NC, PathBuf::from(".")).unwrap();
        cfg.frequency_curve.push((0.0, f64::NAN, 0.0));
        cfg.frequency_curve.push((0.0, -1.0, 0.0));
        cfg.frequency_curve.push((0.0, f64::INFINITY, 0.0));
        let freqs = expand_frequency_curve(&cfg);
        assert!(freqs.iter().all(|f| f.is_finite() && *f > 0.0));
        assert!(freqs.windows(2).all(|w| w[0] <= w[1]));
    }

    fn sample_element() -> Element {
        Element {
            connectivity: vec![0, 1, 2],
            element_type: crate::core::types::ElementType::Tri3,
            property: ElementProperty::Surface,
            normal: [0.0; 3],
            node_normals: vec![[0.0; 3]; 3],
            center: [0.0; 3],
            area: 0.0,
            boundary_condition: BoundaryCondition::Velocity(vec![Complex64::new(0.0, 0.0)]),
            group: 0,
            dof_addresses: vec![0],
        }
    }
}
