//! VTK Unstructured Grid (.vtu) writer.
//!
//! Parameterized over the [`MeshView`] trait so both surface meshes
//! (`sonium_xem::types::RoomMesh`) and volumetric FEM meshes
//! (`sonium_fem::mesh::Mesh`) feed the same writer without conversion.
//!
//! Real or complex scalar and 3-component vector fields are supported at both
//! mesh points and cell centres.
//!
//! # Fallible vs panicking field writers
//!
//! Every field writer comes in two variants: `try_add_*`, which returns
//! [`VtuError`] on a name/arity/non-finite mismatch, and the bare `add_*`,
//! which panics with the same error. The bare forms exist because downstream
//! writer call sites (propellers, structural previews, speaker field export)
//! construct matching buffers by construction; any other caller — and all
//! library code on a hot path — should prefer the `try_add_*` twins.

use num_complex::Complex64;
use std::path::Path;
use vtkio::Vtk;
use vtkio::model::*;

/// Parse and validate a serialized VTU file produced by [`VtuWriter`].
///
/// Keeping this check next to the writer makes output-path tests independent
/// of the caller's knowledge of `vtkio`'s internal model. The parser also
/// catches truncated XML, invalid base64/appended data, and non-VTU datasets.
pub fn validate_vtu_file(path: impl AsRef<Path>) -> std::io::Result<()> {
    let file = std::fs::File::open(path)?;
    let vtk = Vtk::parse_xml(std::io::BufReader::new(file))
        .map_err(|error| std::io::Error::new(std::io::ErrorKind::InvalidData, error.to_string()))?;
    match vtk.data {
        DataSet::UnstructuredGrid { pieces, .. } if pieces.len() == 1 => Ok(()),
        DataSet::UnstructuredGrid { pieces, .. } => Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!(
                "VTU file must contain exactly one piece, found {}",
                pieces.len()
            ),
        )),
        _ => Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "VTU file does not contain an unstructured grid",
        )),
    }
}

/// Element / cell topology supported by [`VtuWriter`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CellKind {
    Triangle,
    Quad,
    Tetra,
    Hexahedron,
}

/// Checked admission failure for a VTU mesh or field.
///
/// The historical `add_*` methods remain available for compatibility and
/// panic on invalid input as they did before. New callers that handle
/// untrusted solver/output data should use the `try_add_*` methods and the
/// writer's checked serialization path instead.
#[derive(Debug, Clone, PartialEq)]
pub enum VtuError {
    MeshValidation(String),
    NodeCountMismatch {
        declared: usize,
        actual: usize,
    },
    CellCountMismatch {
        declared: usize,
        actual: usize,
    },
    NonFiniteNode {
        node: usize,
        component: usize,
    },
    InvalidCellIndex {
        cell: usize,
        node: u64,
        node_count: usize,
    },
    InvalidCellArity {
        cell: usize,
        kind: CellKind,
        actual: usize,
    },
    EmptyFieldName,
    InvalidFieldName {
        name: String,
    },
    FieldLength {
        location: &'static str,
        name: String,
        expected: usize,
        actual: usize,
    },
    NonFiniteField {
        location: &'static str,
        name: String,
        index: usize,
        component: usize,
    },
}

impl std::fmt::Display for VtuError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MeshValidation(message) => write!(f, "invalid VTU mesh: {message}"),
            Self::NodeCountMismatch { declared, actual } => {
                write!(f, "mesh declares {declared} nodes but yielded {actual}")
            }
            Self::CellCountMismatch { declared, actual } => {
                write!(f, "mesh declares {declared} cells but yielded {actual}")
            }
            Self::NonFiniteNode { node, component } => {
                write!(f, "mesh node {node} component {component} is non-finite")
            }
            Self::InvalidCellIndex {
                cell,
                node,
                node_count,
            } => write!(
                f,
                "cell {cell} references node {node}, but mesh has {node_count} nodes"
            ),
            Self::InvalidCellArity { cell, kind, actual } => {
                write!(f, "cell {cell} of kind {kind:?} has invalid arity {actual}")
            }
            Self::EmptyFieldName => f.write_str("VTU field name must not be empty"),
            Self::InvalidFieldName { name } => {
                write!(
                    f,
                    "VTU field name {name:?} contains an XML-invalid character"
                )
            }
            Self::FieldLength {
                location,
                name,
                expected,
                actual,
            } => write!(
                f,
                "{location} field {name:?} has {actual} values, expected {expected}"
            ),
            Self::NonFiniteField {
                location,
                name,
                index,
                component,
            } => write!(
                f,
                "{location} field {name:?} value {index} component {component} is non-finite"
            ),
        }
    }
}

impl std::error::Error for VtuError {}

impl From<VtuError> for std::io::Error {
    fn from(error: VtuError) -> Self {
        Self::new(std::io::ErrorKind::InvalidInput, error)
    }
}

impl CellKind {
    fn to_vtk(self) -> CellType {
        match self {
            CellKind::Triangle => CellType::Triangle,
            CellKind::Quad => CellType::Quad,
            CellKind::Tetra => CellType::Tetra,
            CellKind::Hexahedron => CellType::Hexahedron,
        }
    }
}

/// Mesh geometry exposed to the writer. Implementors yield node coordinates
/// and an iterator of `(kind, vertex_indices)` for each cell.
pub trait MeshView {
    fn num_nodes(&self) -> usize;
    fn num_cells(&self) -> usize;
    /// Iterate node positions as `[x, y, z]`.
    fn nodes(&self) -> Box<dyn Iterator<Item = [f64; 3]> + '_>;
    /// Iterate cells, returning vertex node indices for each.
    fn cells(&self) -> Box<dyn Iterator<Item = (CellKind, Vec<u64>)> + '_>;
    /// Validate that the mesh can be written (e.g. all cell types are supported).
    fn validate(&self) -> Result<(), String> {
        Ok(())
    }
}

enum PointData {
    Scalar { name: String, values: Vec<f64> },
    Vector { name: String, values: Vec<[f64; 3]> },
}

enum CellData {
    Scalar { name: String, values: Vec<f64> },
    Vector { name: String, values: Vec<[f64; 3]> },
}

/// Writer that accumulates point/cell data and serializes a `.vtu`.
pub struct VtuWriter<'m> {
    mesh: &'m dyn MeshView,
    point_data: Vec<PointData>,
    cell_data: Vec<CellData>,
}

impl<'m> VtuWriter<'m> {
    pub fn new(mesh: &'m dyn MeshView) -> Self {
        Self {
            mesh,
            point_data: Vec::new(),
            cell_data: Vec::new(),
        }
    }

    fn validate_field_name(name: &str) -> Result<(), VtuError> {
        if name.is_empty() {
            return Err(VtuError::EmptyFieldName);
        }
        // vtkio writes names into XML attributes. Reject control characters
        // and the five characters that would make an attribute ambiguous.
        if name.chars().any(|character| {
            character.is_control() || matches!(character, '<' | '>' | '&' | '"' | '\'')
        }) {
            return Err(VtuError::InvalidFieldName {
                name: name.to_string(),
            });
        }
        Ok(())
    }

    fn validate_real_field(
        location: &'static str,
        name: &str,
        values: &[f64],
        expected: usize,
    ) -> Result<(), VtuError> {
        Self::validate_field_name(name)?;
        if values.len() != expected {
            return Err(VtuError::FieldLength {
                location,
                name: name.to_string(),
                expected,
                actual: values.len(),
            });
        }
        if let Some(index) = values.iter().position(|value| !value.is_finite()) {
            return Err(VtuError::NonFiniteField {
                location,
                name: name.to_string(),
                index,
                component: 0,
            });
        }
        Ok(())
    }

    fn validate_complex_field(
        location: &'static str,
        name: &str,
        values: &[Complex64],
        expected: usize,
    ) -> Result<(), VtuError> {
        Self::validate_field_name(name)?;
        if values.len() != expected {
            return Err(VtuError::FieldLength {
                location,
                name: name.to_string(),
                expected,
                actual: values.len(),
            });
        }
        if let Some((index, value)) = values
            .iter()
            .enumerate()
            .find(|(_, value)| !value.re.is_finite() || !value.im.is_finite())
        {
            let component = if value.re.is_finite() { 1 } else { 0 };
            return Err(VtuError::NonFiniteField {
                location,
                name: name.to_string(),
                index,
                component,
            });
        }
        Ok(())
    }

    fn validate_real_vector_field(
        location: &'static str,
        name: &str,
        values: &[[f64; 3]],
        expected: usize,
    ) -> Result<(), VtuError> {
        Self::validate_field_name(name)?;
        if values.len() != expected {
            return Err(VtuError::FieldLength {
                location,
                name: name.to_string(),
                expected,
                actual: values.len(),
            });
        }
        for (index, value) in values.iter().enumerate() {
            if let Some(component) = value.iter().position(|component| !component.is_finite()) {
                return Err(VtuError::NonFiniteField {
                    location,
                    name: name.to_string(),
                    index,
                    component,
                });
            }
        }
        Ok(())
    }

    fn validate_complex_vector_field(
        location: &'static str,
        name: &str,
        values: &[[Complex64; 3]],
        expected: usize,
    ) -> Result<(), VtuError> {
        Self::validate_field_name(name)?;
        if values.len() != expected {
            return Err(VtuError::FieldLength {
                location,
                name: name.to_string(),
                expected,
                actual: values.len(),
            });
        }
        for (index, value) in values.iter().enumerate() {
            for (component, value) in value.iter().enumerate() {
                if !value.re.is_finite() || !value.im.is_finite() {
                    return Err(VtuError::NonFiniteField {
                        location,
                        name: name.to_string(),
                        index,
                        component,
                    });
                }
            }
        }
        Ok(())
    }

    /// Validate mesh geometry and connectivity before building a VTU object.
    pub fn validate(&self) -> Result<(), VtuError> {
        self.mesh.validate().map_err(VtuError::MeshValidation)?;
        let nodes: Vec<[f64; 3]> = self.mesh.nodes().collect();
        if nodes.len() != self.mesh.num_nodes() {
            return Err(VtuError::NodeCountMismatch {
                declared: self.mesh.num_nodes(),
                actual: nodes.len(),
            });
        }
        for (node, point) in nodes.iter().enumerate() {
            if let Some(component) = point.iter().position(|value| !value.is_finite()) {
                return Err(VtuError::NonFiniteNode { node, component });
            }
        }
        let cells: Vec<(CellKind, Vec<u64>)> = self.mesh.cells().collect();
        if cells.len() != self.mesh.num_cells() {
            return Err(VtuError::CellCountMismatch {
                declared: self.mesh.num_cells(),
                actual: cells.len(),
            });
        }
        for (cell, (kind, vertices)) in cells.iter().enumerate() {
            let valid_arity = match kind {
                CellKind::Triangle => vertices.len() == 3,
                CellKind::Quad => vertices.len() == 4,
                CellKind::Tetra => vertices.len() == 4,
                CellKind::Hexahedron => vertices.len() == 8,
            };
            if !valid_arity {
                return Err(VtuError::InvalidCellArity {
                    cell,
                    kind: *kind,
                    actual: vertices.len(),
                });
            }
            if let Some(&node) = vertices.iter().find(|&&node| node >= nodes.len() as u64) {
                return Err(VtuError::InvalidCellIndex {
                    cell,
                    node,
                    node_count: nodes.len(),
                });
            }
        }
        Ok(())
    }

    pub fn try_add_point_scalar(&mut self, name: &str, values: &[f64]) -> Result<(), VtuError> {
        Self::validate_real_field("point", name, values, self.mesh.num_nodes())?;
        self.point_data.push(PointData::Scalar {
            name: name.to_string(),
            values: values.to_vec(),
        });
        Ok(())
    }

    pub fn add_point_scalar(&mut self, name: &str, values: &[f64]) {
        self.try_add_point_scalar(name, values)
            .unwrap_or_else(|error| panic!("{error}"));
    }

    /// Add a complex-valued scalar; emits `<name>_magnitude`, `<name>_real`,
    /// `<name>_imag` fields.
    pub fn add_point_scalar_complex(&mut self, name: &str, values: &[Complex64]) {
        self.try_add_point_scalar_complex(name, values)
            .unwrap_or_else(|error| panic!("{error}"));
    }

    pub fn try_add_point_scalar_complex(
        &mut self,
        name: &str,
        values: &[Complex64],
    ) -> Result<(), VtuError> {
        Self::validate_complex_field("point", name, values, self.mesh.num_nodes())?;
        let magnitude: Vec<f64> = values.iter().map(|v| v.norm()).collect();
        let real: Vec<f64> = values.iter().map(|v| v.re).collect();
        let imag: Vec<f64> = values.iter().map(|v| v.im).collect();
        if magnitude.iter().any(|value| !value.is_finite()) {
            return Err(VtuError::NonFiniteField {
                location: "point",
                name: name.to_string(),
                index: magnitude
                    .iter()
                    .position(|value| !value.is_finite())
                    .unwrap_or(0),
                component: 0,
            });
        }
        self.point_data.push(PointData::Scalar {
            name: format!("{name}_magnitude"),
            values: magnitude,
        });
        self.point_data.push(PointData::Scalar {
            name: format!("{name}_real"),
            values: real,
        });
        self.point_data.push(PointData::Scalar {
            name: format!("{name}_imag"),
            values: imag,
        });
        Ok(())
    }

    pub fn add_point_vector(&mut self, name: &str, vx: &[f64], vy: &[f64], vz: &[f64]) {
        self.try_add_point_vector(name, vx, vy, vz)
            .unwrap_or_else(|error| panic!("{error}"));
    }

    pub fn try_add_point_vector(
        &mut self,
        name: &str,
        vx: &[f64],
        vy: &[f64],
        vz: &[f64],
    ) -> Result<(), VtuError> {
        let n = self.mesh.num_nodes();
        if vx.len() != n {
            return Err(VtuError::FieldLength {
                location: "point",
                name: name.to_string(),
                expected: n,
                actual: vx.len(),
            });
        }
        if vy.len() != n {
            return Err(VtuError::FieldLength {
                location: "point",
                name: name.to_string(),
                expected: n,
                actual: vy.len(),
            });
        }
        if vz.len() != n {
            return Err(VtuError::FieldLength {
                location: "point",
                name: name.to_string(),
                expected: n,
                actual: vz.len(),
            });
        }
        let values: Vec<[f64; 3]> = (0..n).map(|i| [vx[i], vy[i], vz[i]]).collect();
        Self::validate_real_vector_field("point", name, &values, n)?;
        self.point_data.push(PointData::Vector {
            name: name.to_string(),
            values,
        });
        Ok(())
    }

    /// Add a complex vector field as magnitude scalar plus real/imaginary vectors.
    pub fn add_point_vector_complex(&mut self, name: &str, values: &[[Complex64; 3]]) {
        self.try_add_point_vector_complex(name, values)
            .unwrap_or_else(|error| panic!("{error}"));
    }

    pub fn try_add_point_vector_complex(
        &mut self,
        name: &str,
        values: &[[Complex64; 3]],
    ) -> Result<(), VtuError> {
        Self::validate_complex_vector_field("point", name, values, self.mesh.num_nodes())?;
        let magnitude: Vec<f64> = values
            .iter()
            .map(|value| {
                value
                    .iter()
                    .map(|component| component.norm_sqr())
                    .sum::<f64>()
                    .sqrt()
            })
            .collect();
        if let Some(index) = magnitude.iter().position(|value| !value.is_finite()) {
            return Err(VtuError::NonFiniteField {
                location: "point",
                name: name.to_string(),
                index,
                component: 0,
            });
        }
        let real: Vec<[f64; 3]> = values
            .iter()
            .map(|value| [value[0].re, value[1].re, value[2].re])
            .collect();
        let imag: Vec<[f64; 3]> = values
            .iter()
            .map(|value| [value[0].im, value[1].im, value[2].im])
            .collect();
        self.point_data.push(PointData::Scalar {
            name: format!("{name}_magnitude"),
            values: magnitude,
        });
        self.point_data.push(PointData::Vector {
            name: format!("{name}_real"),
            values: real,
        });
        self.point_data.push(PointData::Vector {
            name: format!("{name}_imag"),
            values: imag,
        });
        Ok(())
    }

    pub fn add_cell_scalar(&mut self, name: &str, values: &[f64]) {
        self.try_add_cell_scalar(name, values)
            .unwrap_or_else(|error| panic!("{error}"));
    }

    pub fn try_add_cell_scalar(&mut self, name: &str, values: &[f64]) -> Result<(), VtuError> {
        Self::validate_real_field("cell", name, values, self.mesh.num_cells())?;
        self.cell_data.push(CellData::Scalar {
            name: name.to_string(),
            values: values.to_vec(),
        });
        Ok(())
    }

    /// Add a complex cell scalar as magnitude, real, and imaginary arrays.
    pub fn add_cell_scalar_complex(&mut self, name: &str, values: &[Complex64]) {
        self.try_add_cell_scalar_complex(name, values)
            .unwrap_or_else(|error| panic!("{error}"));
    }

    pub fn try_add_cell_scalar_complex(
        &mut self,
        name: &str,
        values: &[Complex64],
    ) -> Result<(), VtuError> {
        Self::validate_complex_field("cell", name, values, self.mesh.num_cells())?;
        let magnitude: Vec<f64> = values.iter().map(|value| value.norm()).collect();
        if let Some(index) = magnitude.iter().position(|value| !value.is_finite()) {
            return Err(VtuError::NonFiniteField {
                location: "cell",
                name: name.to_string(),
                index,
                component: 0,
            });
        }
        self.cell_data.push(CellData::Scalar {
            name: format!("{name}_magnitude"),
            values: magnitude,
        });
        self.cell_data.push(CellData::Scalar {
            name: format!("{name}_real"),
            values: values.iter().map(|value| value.re).collect(),
        });
        self.cell_data.push(CellData::Scalar {
            name: format!("{name}_imag"),
            values: values.iter().map(|value| value.im).collect(),
        });
        Ok(())
    }

    pub fn add_cell_vector(&mut self, name: &str, values: &[[f64; 3]]) {
        self.try_add_cell_vector(name, values)
            .unwrap_or_else(|error| panic!("{error}"));
    }

    pub fn try_add_cell_vector(&mut self, name: &str, values: &[[f64; 3]]) -> Result<(), VtuError> {
        Self::validate_real_vector_field("cell", name, values, self.mesh.num_cells())?;
        self.cell_data.push(CellData::Vector {
            name: name.to_string(),
            values: values.to_vec(),
        });
        Ok(())
    }

    /// Add a complex cell vector as magnitude scalar plus real/imaginary vectors.
    pub fn add_cell_vector_complex(&mut self, name: &str, values: &[[Complex64; 3]]) {
        self.try_add_cell_vector_complex(name, values)
            .unwrap_or_else(|error| panic!("{error}"));
    }

    pub fn try_add_cell_vector_complex(
        &mut self,
        name: &str,
        values: &[[Complex64; 3]],
    ) -> Result<(), VtuError> {
        Self::validate_complex_vector_field("cell", name, values, self.mesh.num_cells())?;
        let magnitude: Vec<f64> = values
            .iter()
            .map(|value| {
                value
                    .iter()
                    .map(|component| component.norm_sqr())
                    .sum::<f64>()
                    .sqrt()
            })
            .collect();
        if let Some(index) = magnitude.iter().position(|value| !value.is_finite()) {
            return Err(VtuError::NonFiniteField {
                location: "cell",
                name: name.to_string(),
                index,
                component: 0,
            });
        }
        let real: Vec<[f64; 3]> = values
            .iter()
            .map(|value| [value[0].re, value[1].re, value[2].re])
            .collect();
        let imag: Vec<[f64; 3]> = values
            .iter()
            .map(|value| [value[0].im, value[1].im, value[2].im])
            .collect();
        self.cell_data.push(CellData::Scalar {
            name: format!("{name}_magnitude"),
            values: magnitude,
        });
        self.cell_data.push(CellData::Vector {
            name: format!("{name}_real"),
            values: real,
        });
        self.cell_data.push(CellData::Vector {
            name: format!("{name}_imag"),
            values: imag,
        });
        Ok(())
    }

    /// Write to disk; extension determines format (`.vtu` XML, `.vtk` legacy).
    pub fn write(&self, path: impl AsRef<Path>) -> std::io::Result<()> {
        self.validate().map_err(std::io::Error::from)?;
        let vtk = self.build_vtk();
        vtk.export(path)
            .map_err(|e| std::io::Error::other(e.to_string()))
    }

    pub fn write_to<W: std::io::Write>(&self, w: &mut W) -> std::io::Result<()> {
        self.validate().map_err(std::io::Error::from)?;
        let vtk = self.build_vtk();
        vtk.write_xml(w)
            .map_err(|e| std::io::Error::other(e.to_string()))
    }

    pub fn write_ascii(&self, path: impl AsRef<Path>) -> std::io::Result<()> {
        self.validate().map_err(std::io::Error::from)?;
        let vtk = self.build_vtk();
        vtk.export_ascii(path)
            .map_err(|e| std::io::Error::other(e.to_string()))
    }

    fn build_vtk(&self) -> Vtk {
        let points: Vec<f64> = self.mesh.nodes().flat_map(|p| [p[0], p[1], p[2]]).collect();

        let mut connectivity: Vec<u64> = Vec::new();
        let mut offsets: Vec<u64> = Vec::new();
        let mut cell_types: Vec<CellType> = Vec::new();
        let mut offset: u64 = 0;

        for (kind, verts) in self.mesh.cells() {
            offset += verts.len() as u64;
            connectivity.extend(verts);
            offsets.push(offset);
            cell_types.push(kind.to_vtk());
        }

        let mut point_attrs: Vec<Attribute> = Vec::new();
        for field in &self.point_data {
            match field {
                PointData::Scalar { name, values } => {
                    point_attrs.push(Attribute::DataArray(
                        DataArray::scalars(name.clone(), 1)
                            .with_data(values.iter().map(|&v| v as f32).collect::<Vec<f32>>()),
                    ));
                }
                PointData::Vector { name, values } => {
                    let flat: Vec<f32> = values
                        .iter()
                        .flat_map(|v| [v[0] as f32, v[1] as f32, v[2] as f32])
                        .collect();
                    point_attrs.push(Attribute::DataArray(
                        DataArray::vectors(name.clone()).with_data(flat),
                    ));
                }
            }
        }

        let mut cell_attrs: Vec<Attribute> = Vec::new();
        for field in &self.cell_data {
            match field {
                CellData::Scalar { name, values } => {
                    cell_attrs.push(Attribute::DataArray(
                        DataArray::scalars(name.clone(), 1)
                            .with_data(values.iter().map(|&v| v as f32).collect::<Vec<f32>>()),
                    ));
                }
                CellData::Vector { name, values } => {
                    let flat: Vec<f32> = values
                        .iter()
                        .flat_map(|value| [value[0] as f32, value[1] as f32, value[2] as f32])
                        .collect();
                    cell_attrs.push(Attribute::DataArray(
                        DataArray::vectors(name.clone()).with_data(flat),
                    ));
                }
            }
        }

        let piece = UnstructuredGridPiece {
            points: IOBuffer::F64(points),
            cells: Cells {
                cell_verts: VertexNumbers::XML {
                    connectivity,
                    offsets,
                },
                types: cell_types,
            },
            data: Attributes {
                point: point_attrs,
                cell: cell_attrs,
            },
        };

        Vtk {
            version: Version::new(),
            title: String::from("Sonium"),
            byte_order: ByteOrder::LittleEndian,
            file_path: None,
            data: DataSet::inline(piece),
        }
    }
}

// ---------------------------------------------------------------------------
// MeshView impl for sonium_xem::types::RoomMesh
// ---------------------------------------------------------------------------

impl MeshView for crate::types::RoomMesh {
    fn num_nodes(&self) -> usize {
        self.nodes.len()
    }

    fn num_cells(&self) -> usize {
        self.elements.len()
    }

    fn nodes(&self) -> Box<dyn Iterator<Item = [f64; 3]> + '_> {
        Box::new(self.nodes.iter().map(|p| [p.x, p.y, p.z]))
    }

    fn cells(&self) -> Box<dyn Iterator<Item = (CellKind, Vec<u64>)> + '_> {
        Box::new(self.elements.iter().map(|e| {
            let kind = match e.nodes.len() {
                3 => CellKind::Triangle,
                4 => CellKind::Quad,
                // Fallback: treat unsupported cells as triangles (will be caught by validate)
                _ => CellKind::Triangle,
            };
            (kind, e.nodes.iter().map(|&i| i as u64).collect())
        }))
    }

    fn validate(&self) -> Result<(), String> {
        for (i, e) in self.elements.iter().enumerate() {
            match e.nodes.len() {
                3 | 4 => {}
                n => {
                    return Err(format!(
                        "RoomMesh element {i} with {n} nodes is not supported by VtuWriter"
                    ));
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{Point3D, RoomMesh, SurfaceElement};

    fn tiny_mesh() -> RoomMesh {
        RoomMesh {
            nodes: vec![
                Point3D::new(0.0, 0.0, 0.0),
                Point3D::new(1.0, 0.0, 0.0),
                Point3D::new(1.0, 1.0, 0.0),
                Point3D::new(0.0, 1.0, 0.0),
            ],
            elements: vec![
                SurfaceElement::triangle(0, 1, 2),
                SurfaceElement::triangle(0, 2, 3),
            ],
        }
    }

    #[test]
    fn roundtrip_roommesh_scalar() {
        let mesh = tiny_mesh();
        let mut w = VtuWriter::new(&mesh);
        w.add_point_scalar("temperature", &[1.0, 2.0, 3.0, 4.0]);
        let mut buf = Vec::new();
        w.write_to(&mut buf).unwrap();
        let s = String::from_utf8_lossy(&buf);
        assert!(s.contains("UnstructuredGrid"));
        assert!(s.contains("temperature"));
    }

    #[test]
    fn complex_field_emits_three_arrays() {
        let mesh = tiny_mesh();
        let field: Vec<Complex64> = (0..4).map(|i| Complex64::new(i as f64, 0.5)).collect();
        let mut w = VtuWriter::new(&mesh);
        w.add_point_scalar_complex("pressure", &field);
        let mut buf = Vec::new();
        w.write_to(&mut buf).unwrap();
        let s = String::from_utf8_lossy(&buf);
        assert!(s.contains("pressure_magnitude"));
        assert!(s.contains("pressure_real"));
        assert!(s.contains("pressure_imag"));
    }

    #[test]
    fn cell_data_roundtrip() {
        let mesh = tiny_mesh();
        let mut w = VtuWriter::new(&mesh);
        w.add_cell_scalar("material_id", &[1.0, 2.0]);
        let mut buf = Vec::new();
        w.write_to(&mut buf).unwrap();
        let s = String::from_utf8_lossy(&buf);
        assert!(s.contains("material_id"));
    }

    #[test]
    fn complex_point_and_cell_vectors_emit_typed_arrays() {
        let mesh = tiny_mesh();
        let point_vectors = vec![[Complex64::new(1.0, 0.5); 3]; mesh.nodes.len()];
        let cell_vectors = vec![[Complex64::new(2.0, -0.25); 3]; mesh.elements.len()];
        let cell_scalars = vec![Complex64::new(3.0, 4.0); mesh.elements.len()];
        let mut writer = VtuWriter::new(&mesh);
        writer.add_point_vector_complex("displacement", &point_vectors);
        writer.add_cell_vector_complex("magnetic_flux_density", &cell_vectors);
        writer.add_cell_scalar_complex("eddy_loss_density", &cell_scalars);

        let mut buffer = Vec::new();
        writer.write_to(&mut buffer).unwrap();
        let xml = String::from_utf8_lossy(&buffer);
        for name in [
            "displacement_magnitude",
            "displacement_real",
            "displacement_imag",
            "magnetic_flux_density_magnitude",
            "magnetic_flux_density_real",
            "magnetic_flux_density_imag",
            "eddy_loss_density_magnitude",
            "eddy_loss_density_real",
            "eddy_loss_density_imag",
        ] {
            assert!(xml.contains(name), "missing VTU array {name}");
        }
    }

    #[test]
    fn ascii_export_to_disk() {
        let mesh = tiny_mesh();
        let mut w = VtuWriter::new(&mesh);
        w.add_point_scalar("p", &[0.0; 4]);
        let path = std::env::temp_dir().join("sonium_xem_vtu_test.vtu");
        w.write_ascii(&path).unwrap();
        let s = std::fs::read_to_string(&path).unwrap();
        assert!(s.contains("UNSTRUCTURED_GRID"));
    }

    #[test]
    fn checked_field_admission_rejects_bad_names_lengths_and_values() {
        let mesh = tiny_mesh();
        let mut writer = VtuWriter::new(&mesh);
        assert!(matches!(
            writer.try_add_point_scalar("", &[0.0; 4]),
            Err(VtuError::EmptyFieldName)
        ));
        assert!(matches!(
            writer.try_add_point_scalar("bad<name", &[0.0; 4]),
            Err(VtuError::InvalidFieldName { .. })
        ));
        assert!(matches!(
            writer.try_add_point_scalar("temperature", &[0.0; 3]),
            Err(VtuError::FieldLength { .. })
        ));
        assert!(matches!(
            writer.try_add_point_scalar("temperature", &[0.0, f64::NAN, 0.0, 0.0]),
            Err(VtuError::NonFiniteField { .. })
        ));
        assert!(matches!(
            writer.try_add_point_vector(
                "velocity",
                &[0.0; 4],
                &[0.0, f64::INFINITY, 0.0, 0.0],
                &[0.0; 4]
            ),
            Err(VtuError::NonFiniteField { .. })
        ));
    }

    #[test]
    fn checked_mesh_admission_rejects_out_of_range_connectivity() {
        let mut mesh = tiny_mesh();
        mesh.elements[0].nodes[0] = 99;
        let writer = VtuWriter::new(&mesh);
        assert!(matches!(
            writer.validate(),
            Err(VtuError::InvalidCellIndex { cell: 0, .. })
        ));
        let mut output = Vec::new();
        let error = writer
            .write_to(&mut output)
            .expect_err("invalid connectivity must fail");
        assert_eq!(error.kind(), std::io::ErrorKind::InvalidInput);
        assert!(output.is_empty());
    }

    #[test]
    fn vtkio_parser_reads_serialized_topology_and_arrays() {
        let mesh = tiny_mesh();
        let mut writer = VtuWriter::new(&mesh);
        writer
            .try_add_point_scalar("temperature", &[1.0, 2.0, 3.0, 4.0])
            .unwrap();
        writer.try_add_cell_scalar("region", &[7.0, 8.0]).unwrap();
        let mut output = Vec::new();
        writer.write_to(&mut output).unwrap();

        let parsed = Vtk::parse_xml(std::io::BufReader::new(output.as_slice())).unwrap();
        let DataSet::UnstructuredGrid { pieces, .. } = parsed.data else {
            panic!("VTU parser returned a non-unstructured data set");
        };
        assert_eq!(pieces.len(), 1);
        let Piece::Inline(piece) = &pieces[0] else {
            panic!("serialized VTU unexpectedly references an external piece");
        };
        assert_eq!(piece.cells.types.len(), 2);
        assert_eq!(piece.cells.cell_verts.num_cells(), 2);
        assert!(
            piece
                .data
                .point
                .iter()
                .any(|attribute| format!("{attribute:?}").contains("temperature"))
        );
        assert!(
            piece
                .data
                .cell
                .iter()
                .any(|attribute| format!("{attribute:?}").contains("region"))
        );
    }
}
