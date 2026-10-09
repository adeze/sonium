//! Speaker-study VTU/PVD export with explicit field associations.

use crate::conventions::{ArtifactProvenance, ConventionError, StudyMetadata};
use num_complex::Complex64;
use serde::{Deserialize, Serialize};
use sonium_fem::io::vtk::{PvdCollection, VtuWriter};
use sonium_fem::mesh::{Mesh, PhysicalGroup};
use std::path::{Path, PathBuf};

/// Canonical field names used by speaker study exporters and QA tooling.
pub mod field_names {
    pub const REGION_MARKER: &str = "region_marker";
    pub const MATERIAL_ID: &str = "material_id";
    pub const CONDUCTIVITY: &str = "conductivity_s_per_m";
    pub const RELATIVE_PERMEABILITY: &str = "relative_permeability";
    pub const MAGNETIC_POTENTIAL: &str = "magnetic_potential";
    pub const MAGNETIC_FLUX_DENSITY: &str = "magnetic_flux_density";
    pub const MAGNETIC_FLUX_DENSITY_MAGNITUDE: &str = "magnetic_flux_density_magnitude_t";
    pub const MAGNETIC_FIELD_STRENGTH: &str = "magnetic_field_strength";
    pub const SOURCE_CURRENT_DENSITY: &str = "source_current_density";
    pub const EDDY_CURRENT_DENSITY: &str = "eddy_current_density";
    pub const EDDY_LOSS_DENSITY: &str = "eddy_loss_density";
    pub const LORENTZ_FORCE_DENSITY: &str = "lorentz_force_density";
    pub const DISPLACEMENT: &str = "displacement";
    pub const VELOCITY: &str = "velocity";
    pub const STRESS: &str = "stress";
    pub const VON_MISES_STRESS: &str = "von_mises_stress";
    pub const ACOUSTIC_PRESSURE: &str = "acoustic_pressure";
    pub const SOUND_PRESSURE_LEVEL: &str = "sound_pressure_level_db";
    pub const ACOUSTIC_INTENSITY: &str = "acoustic_intensity";
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FieldAssociation {
    Point,
    Cell,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FieldValueKind {
    RealScalar,
    ComplexScalar,
    RealVector,
    ComplexVector,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProjectionProvenance {
    ElementCentroid,
    QuadratureAverage,
    L2Projection,
}

/// A vector field deliberately projected to one value per mesh cell.
///
/// H(curl), discontinuous, and quadrature fields must pass through this type;
/// the exporter never treats raw field-space DOFs as Cartesian nodal vectors.
#[derive(Debug, Clone, PartialEq)]
pub struct ProjectedCellVector<T> {
    values: Vec<[T; 3]>,
    provenance: ProjectionProvenance,
}

impl<T> ProjectedCellVector<T> {
    pub fn new(values: Vec<[T; 3]>, provenance: ProjectionProvenance) -> Self {
        Self { values, provenance }
    }

    pub fn values(&self) -> &[[T; 3]] {
        &self.values
    }

    pub fn provenance(&self) -> ProjectionProvenance {
        self.provenance
    }
}

impl ProjectedCellVector<Complex64> {
    /// Arithmetic cell averages from per-cell quadrature samples.
    pub fn from_quadrature_average(
        samples: &[Vec<[Complex64; 3]>],
    ) -> Result<Self, FieldExportError> {
        let mut values = Vec::with_capacity(samples.len());
        for (cell_index, cell_samples) in samples.iter().enumerate() {
            if cell_samples.is_empty() {
                return Err(FieldExportError::EmptyQuadratureCell { cell_index });
            }
            let mut sum = [Complex64::new(0.0, 0.0); 3];
            for sample in cell_samples {
                for component in 0..3 {
                    sum[component] += sample[component];
                }
            }
            let scale = 1.0 / cell_samples.len() as f64;
            values.push([sum[0] * scale, sum[1] * scale, sum[2] * scale]);
        }
        Ok(Self::new(values, ProjectionProvenance::QuadratureAverage))
    }
}

#[derive(Debug, Clone)]
enum FieldValues {
    PointRealScalar(Vec<f64>),
    PointComplexScalar(Vec<Complex64>),
    PointRealVector(Vec<[f64; 3]>),
    PointComplexVector(Vec<[Complex64; 3]>),
    CellRealScalar(Vec<f64>),
    CellComplexScalar(Vec<Complex64>),
    CellRealVector(ProjectedCellVector<f64>),
    CellComplexVector(ProjectedCellVector<Complex64>),
}

#[derive(Debug, Clone)]
struct NamedField {
    name: String,
    values: FieldValues,
}

#[derive(Debug, Clone, Default)]
pub struct SpeakerFieldBundle {
    fields: Vec<NamedField>,
}

impl SpeakerFieldBundle {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_point_real_scalar(
        &mut self,
        name: impl Into<String>,
        values: Vec<f64>,
    ) -> Result<(), FieldExportError> {
        self.insert(name, FieldValues::PointRealScalar(values))
    }

    pub fn add_point_complex_scalar(
        &mut self,
        name: impl Into<String>,
        values: Vec<Complex64>,
    ) -> Result<(), FieldExportError> {
        self.insert(name, FieldValues::PointComplexScalar(values))
    }

    pub fn add_point_real_vector(
        &mut self,
        name: impl Into<String>,
        values: Vec<[f64; 3]>,
    ) -> Result<(), FieldExportError> {
        self.insert(name, FieldValues::PointRealVector(values))
    }

    pub fn add_point_complex_vector(
        &mut self,
        name: impl Into<String>,
        values: Vec<[Complex64; 3]>,
    ) -> Result<(), FieldExportError> {
        self.insert(name, FieldValues::PointComplexVector(values))
    }

    pub fn add_cell_real_scalar(
        &mut self,
        name: impl Into<String>,
        values: Vec<f64>,
    ) -> Result<(), FieldExportError> {
        self.insert(name, FieldValues::CellRealScalar(values))
    }

    pub fn add_cell_complex_scalar(
        &mut self,
        name: impl Into<String>,
        values: Vec<Complex64>,
    ) -> Result<(), FieldExportError> {
        self.insert(name, FieldValues::CellComplexScalar(values))
    }

    pub fn add_cell_real_vector(
        &mut self,
        name: impl Into<String>,
        values: ProjectedCellVector<f64>,
    ) -> Result<(), FieldExportError> {
        self.insert(name, FieldValues::CellRealVector(values))
    }

    pub fn add_cell_complex_vector(
        &mut self,
        name: impl Into<String>,
        values: ProjectedCellVector<Complex64>,
    ) -> Result<(), FieldExportError> {
        self.insert(name, FieldValues::CellComplexVector(values))
    }

    /// Expand a complex symmetric stress tensor into six named cell scalars.
    pub fn add_cell_complex_symmetric_tensor(
        &mut self,
        name: &str,
        values: Vec<[Complex64; 6]>,
    ) -> Result<(), FieldExportError> {
        let suffixes = ["xx", "yy", "zz", "xy", "yz", "xz"];
        for suffix in suffixes {
            let component_name = format!("{name}_{suffix}");
            if self.fields.iter().any(|field| field.name == component_name) {
                return Err(FieldExportError::DuplicateField(component_name));
            }
        }
        for (component, suffix) in suffixes.into_iter().enumerate() {
            self.add_cell_complex_scalar(
                format!("{name}_{suffix}"),
                values.iter().map(|value| value[component]).collect(),
            )?;
        }
        Ok(())
    }

    fn insert(
        &mut self,
        name: impl Into<String>,
        values: FieldValues,
    ) -> Result<(), FieldExportError> {
        let name = name.into();
        if name.trim().is_empty() {
            return Err(FieldExportError::BlankFieldName);
        }
        if self.fields.iter().any(|field| field.name == name) {
            return Err(FieldExportError::DuplicateField(name));
        }
        self.fields.push(NamedField { name, values });
        Ok(())
    }

    fn validate(&self, mesh: &Mesh) -> Result<(), FieldExportError> {
        for field in &self.fields {
            let (actual, expected) = match &field.values {
                FieldValues::PointRealScalar(values) => (values.len(), mesh.num_nodes()),
                FieldValues::PointComplexScalar(values) => (values.len(), mesh.num_nodes()),
                FieldValues::PointRealVector(values) => (values.len(), mesh.num_nodes()),
                FieldValues::PointComplexVector(values) => (values.len(), mesh.num_nodes()),
                FieldValues::CellRealScalar(values) => (values.len(), mesh.num_elements()),
                FieldValues::CellComplexScalar(values) => (values.len(), mesh.num_elements()),
                FieldValues::CellRealVector(values) => (values.values.len(), mesh.num_elements()),
                FieldValues::CellComplexVector(values) => {
                    (values.values.len(), mesh.num_elements())
                }
            };
            if actual != expected {
                return Err(FieldExportError::LengthMismatch {
                    field: field.name.clone(),
                    expected,
                    actual,
                });
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FieldDescriptor {
    pub name: String,
    pub association: FieldAssociation,
    pub value_kind: FieldValueKind,
    pub projection: Option<ProjectionProvenance>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SpeakerExportManifest {
    pub metadata: StudyMetadata,
    pub physical_groups: Vec<PhysicalGroup>,
    pub fields: Vec<FieldDescriptor>,
    /// Backend/profile/precision of the solve that produced these fields.
    /// Defaults keep older sidecars readable.
    #[serde(default)]
    pub provenance: ArtifactProvenance,
}

pub struct SpeakerFieldExporter<'a> {
    mesh: &'a Mesh,
    metadata: StudyMetadata,
    fields: &'a SpeakerFieldBundle,
    provenance: ArtifactProvenance,
}

impl<'a> SpeakerFieldExporter<'a> {
    pub fn new(mesh: &'a Mesh, metadata: StudyMetadata, fields: &'a SpeakerFieldBundle) -> Self {
        Self {
            mesh,
            metadata,
            fields,
            provenance: ArtifactProvenance::native_golden(),
        }
    }

    /// Override the default native golden provenance for an explicitly
    /// reduced, fixture, accelerator, or WASM export.
    pub fn with_provenance(mut self, provenance: ArtifactProvenance) -> Self {
        self.provenance = provenance;
        self
    }

    pub fn validate(&self) -> Result<(), FieldExportError> {
        self.metadata.validate()?;
        self.mesh.validate_physical_groups()?;
        self.fields.validate(self.mesh)
    }

    pub fn manifest(&self) -> SpeakerExportManifest {
        let mut fields = vec![FieldDescriptor {
            name: field_names::REGION_MARKER.to_owned(),
            association: FieldAssociation::Cell,
            value_kind: FieldValueKind::RealScalar,
            projection: None,
        }];
        fields.extend(self.fields.fields.iter().map(field_descriptor));
        SpeakerExportManifest {
            metadata: self.metadata.clone(),
            physical_groups: self.mesh.physical_groups.iter().cloned().collect(),
            fields,
            provenance: self.provenance,
        }
    }

    /// Write one VTU snapshot and its `<file>.vtu.json` metadata sidecar.
    pub fn write_snapshot(
        &self,
        vtu_path: impl AsRef<Path>,
    ) -> Result<ExportedSnapshot, FieldExportError> {
        self.validate()?;
        let vtu_path = vtu_path.as_ref().to_path_buf();
        let mut writer = VtuWriter::new(self.mesh);
        writer.add_cell_scalar(
            field_names::REGION_MARKER,
            &self
                .mesh
                .elements
                .iter()
                .map(|element| element.marker as f64)
                .collect::<Vec<_>>(),
        );
        for field in &self.fields.fields {
            add_to_writer(&mut writer, field);
        }
        writer.write(&vtu_path)?;

        let metadata_path = sidecar_path(&vtu_path);
        let manifest = serde_json::to_vec_pretty(&self.manifest())?;
        std::fs::write(&metadata_path, manifest)?;
        Ok(ExportedSnapshot {
            vtu_path,
            metadata_path,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExportedSnapshot {
    pub vtu_path: PathBuf,
    pub metadata_path: PathBuf,
}

pub struct SpeakerPvdCollection {
    directory: PathBuf,
    stem: String,
    collection: PvdCollection,
    next_index: usize,
}

impl SpeakerPvdCollection {
    pub fn new(pvd_path: impl AsRef<Path>) -> Result<Self, FieldExportError> {
        let pvd_path = pvd_path.as_ref();
        let directory = pvd_path
            .parent()
            .unwrap_or_else(|| Path::new("."))
            .to_path_buf();
        let stem = pvd_path
            .file_stem()
            .and_then(|value| value.to_str())
            .filter(|value| !value.is_empty())
            .ok_or(FieldExportError::InvalidPvdPath)?
            .to_owned();
        std::fs::create_dir_all(&directory)?;
        Ok(Self {
            directory,
            stem,
            collection: PvdCollection::new(pvd_path),
            next_index: 0,
        })
    }

    pub fn push(
        &mut self,
        exporter: &SpeakerFieldExporter<'_>,
    ) -> Result<ExportedSnapshot, FieldExportError> {
        let frequency_hz = exporter
            .metadata
            .frequency_hz
            .ok_or(FieldExportError::SweepFrameMissingFrequency)?;
        let filename = format!("{}_{:04}.vtu", self.stem, self.next_index);
        let snapshot = exporter.write_snapshot(self.directory.join(&filename))?;
        self.collection.push(frequency_hz, filename);
        self.next_index += 1;
        Ok(snapshot)
    }

    pub fn finish(self) -> Result<(), FieldExportError> {
        self.collection.finish()?;
        Ok(())
    }
}

fn field_descriptor(field: &NamedField) -> FieldDescriptor {
    let (association, value_kind, projection) = match &field.values {
        FieldValues::PointRealScalar(_) => {
            (FieldAssociation::Point, FieldValueKind::RealScalar, None)
        }
        FieldValues::PointComplexScalar(_) => {
            (FieldAssociation::Point, FieldValueKind::ComplexScalar, None)
        }
        FieldValues::PointRealVector(_) => {
            (FieldAssociation::Point, FieldValueKind::RealVector, None)
        }
        FieldValues::PointComplexVector(_) => {
            (FieldAssociation::Point, FieldValueKind::ComplexVector, None)
        }
        FieldValues::CellRealScalar(_) => {
            (FieldAssociation::Cell, FieldValueKind::RealScalar, None)
        }
        FieldValues::CellComplexScalar(_) => {
            (FieldAssociation::Cell, FieldValueKind::ComplexScalar, None)
        }
        FieldValues::CellRealVector(values) => (
            FieldAssociation::Cell,
            FieldValueKind::RealVector,
            Some(values.provenance),
        ),
        FieldValues::CellComplexVector(values) => (
            FieldAssociation::Cell,
            FieldValueKind::ComplexVector,
            Some(values.provenance),
        ),
    };
    FieldDescriptor {
        name: field.name.clone(),
        association,
        value_kind,
        projection,
    }
}

fn add_to_writer(writer: &mut VtuWriter<'_>, field: &NamedField) {
    match &field.values {
        FieldValues::PointRealScalar(values) => writer.add_point_scalar(&field.name, values),
        FieldValues::PointComplexScalar(values) => {
            writer.add_point_scalar_complex(&field.name, values)
        }
        FieldValues::PointRealVector(values) => {
            let x: Vec<_> = values.iter().map(|value| value[0]).collect();
            let y: Vec<_> = values.iter().map(|value| value[1]).collect();
            let z: Vec<_> = values.iter().map(|value| value[2]).collect();
            writer.add_point_vector(&field.name, &x, &y, &z);
        }
        FieldValues::PointComplexVector(values) => {
            writer.add_point_vector_complex(&field.name, values)
        }
        FieldValues::CellRealScalar(values) => writer.add_cell_scalar(&field.name, values),
        FieldValues::CellComplexScalar(values) => {
            writer.add_cell_scalar_complex(&field.name, values)
        }
        FieldValues::CellRealVector(values) => writer.add_cell_vector(&field.name, values.values()),
        FieldValues::CellComplexVector(values) => {
            writer.add_cell_vector_complex(&field.name, values.values())
        }
    }
}

fn sidecar_path(vtu_path: &Path) -> PathBuf {
    let mut path = vtu_path.to_path_buf();
    path.set_extension("vtu.json");
    path
}

#[derive(Debug, thiserror::Error)]
pub enum FieldExportError {
    #[error(transparent)]
    Convention(#[from] ConventionError),
    #[error(transparent)]
    MeshMetadata(#[from] sonium_fem::mesh::PhysicalGroupError),
    #[error("field name must not be blank")]
    BlankFieldName,
    #[error("field {0:?} is already registered")]
    DuplicateField(String),
    #[error("field {field:?} has {actual} values; expected {expected}")]
    LengthMismatch {
        field: String,
        expected: usize,
        actual: usize,
    },
    #[error("cell {cell_index} has no quadrature samples")]
    EmptyQuadratureCell { cell_index: usize },
    #[error("PVD path must have a non-empty file stem")]
    InvalidPvdPath,
    #[error("every PVD frame requires metadata.frequency_hz")]
    SweepFrameMissingFrequency,
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error(transparent)]
    Json(#[from] serde_json::Error),
}

#[cfg(test)]
mod tests {
    use super::*;
    use sonium_fem::mesh::{ElementType, Point};

    fn marked_triangle() -> Mesh {
        let mut mesh = Mesh::new(2);
        mesh.add_node(Point::new_2d(0.0, 0.0));
        mesh.add_node(Point::new_2d(1.0, 0.0));
        mesh.add_node(Point::new_2d(0.0, 1.0));
        mesh.add_element_with_marker(ElementType::Triangle, vec![0, 1, 2], 10);
        mesh.register_physical_group(2, 10, "acoustic fluid")
            .unwrap();
        mesh
    }

    #[test]
    fn snapshot_contains_typed_fields_regions_and_metadata_sidecar() {
        let mesh = marked_triangle();
        let mut fields = SpeakerFieldBundle::new();
        fields
            .add_point_complex_scalar("acoustic_pressure", vec![Complex64::new(1.0, 0.5); 3])
            .unwrap();
        fields
            .add_point_complex_vector("displacement", vec![[Complex64::new(0.0, 1.0); 3]; 3])
            .unwrap();
        fields
            .add_cell_complex_vector(
                "magnetic_flux_density",
                ProjectedCellVector::new(
                    vec![[Complex64::new(1.0, 0.0); 3]],
                    ProjectionProvenance::ElementCentroid,
                ),
            )
            .unwrap();
        let mut metadata = StudyMetadata::new("export test");
        metadata.frequency_hz = Some(1_000.0);
        metadata.drive_voltage_rms_v = Some(2.83);
        let directory = tempfile::tempdir().unwrap();
        let vtu_path = directory.path().join("speaker.vtu");

        let snapshot = SpeakerFieldExporter::new(&mesh, metadata, &fields)
            .write_snapshot(&vtu_path)
            .unwrap();

        let vtu = std::fs::read_to_string(snapshot.vtu_path).unwrap();
        assert!(vtu.contains("region_marker"));
        assert!(vtu.contains("acoustic_pressure_real"));
        assert!(vtu.contains("displacement_imag"));
        assert!(vtu.contains("magnetic_flux_density_magnitude"));
        let manifest: SpeakerExportManifest =
            serde_json::from_slice(&std::fs::read(snapshot.metadata_path).unwrap()).unwrap();
        assert_eq!(manifest.metadata.frequency_hz, Some(1_000.0));
        assert_eq!(manifest.physical_groups[0].name, "acoustic fluid");
        assert_eq!(
            manifest.provenance,
            ArtifactProvenance::native_golden(),
            "field sidecars must identify native golden exports"
        );
        assert!(manifest.fields.iter().any(|field| {
            field.name == "magnetic_flux_density"
                && field.association == FieldAssociation::Cell
                && field.projection == Some(ProjectionProvenance::ElementCentroid)
        }));
    }

    #[test]
    fn explicit_wasm_preview_provenance_survives_manifest_generation() {
        let mesh = marked_triangle();
        let fields = SpeakerFieldBundle::new();
        let exporter = SpeakerFieldExporter::new(&mesh, StudyMetadata::new("preview"), &fields)
            .with_provenance(ArtifactProvenance::wasm_preview());
        assert_eq!(
            exporter.manifest().provenance,
            ArtifactProvenance::wasm_preview()
        );
    }

    #[test]
    fn pvd_collection_writes_frequency_frames_and_sidecars() {
        let mesh = marked_triangle();
        let fields = SpeakerFieldBundle::new();
        let directory = tempfile::tempdir().unwrap();
        let pvd_path = directory.path().join("speaker_sweep.pvd");
        let mut sweep = SpeakerPvdCollection::new(&pvd_path).unwrap();
        for frequency_hz in [100.0, 200.0] {
            let mut metadata = StudyMetadata::new("sweep");
            metadata.frequency_hz = Some(frequency_hz);
            sweep
                .push(&SpeakerFieldExporter::new(&mesh, metadata, &fields))
                .unwrap();
        }
        sweep.finish().unwrap();

        let pvd = std::fs::read_to_string(pvd_path).unwrap();
        assert!(pvd.contains("timestep=\"100\""));
        assert!(pvd.contains("timestep=\"200\""));
        assert!(
            directory
                .path()
                .join("speaker_sweep_0000.vtu.json")
                .exists()
        );
        assert!(
            directory
                .path()
                .join("speaker_sweep_0001.vtu.json")
                .exists()
        );
    }

    #[test]
    fn quadrature_projection_rejects_empty_cells() {
        let error = ProjectedCellVector::from_quadrature_average(&[Vec::new()])
            .expect_err("empty quadrature samples must fail");
        assert!(matches!(
            error,
            FieldExportError::EmptyQuadratureCell { cell_index: 0 }
        ));
    }

    #[test]
    fn field_lengths_are_validated_before_any_file_is_written() {
        let mesh = marked_triangle();
        let mut fields = SpeakerFieldBundle::new();
        fields.add_point_real_scalar("spl", vec![1.0, 2.0]).unwrap();
        let exporter = SpeakerFieldExporter::new(&mesh, StudyMetadata::new("invalid"), &fields);
        assert!(matches!(
            exporter.validate(),
            Err(FieldExportError::LengthMismatch { .. })
        ));
    }
}
