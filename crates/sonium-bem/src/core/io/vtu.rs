//! VTU helpers for BEM binaries — wrap `sonium_xem::io::vtu::VtuWriter` with
//! convenience functions for the common cases (constant-basis surface
//! pressure, linear-basis nodal pressure).

use crate::core::types::{BasisOrder, Element, ElementType, Mesh};
use num_complex::Complex64;
use sonium_xem::io::vtu::{CellKind, MeshView, VtuWriter};
use sonium_xem::pressure_to_spl;
use std::path::Path;

/// Write surface mesh + complex pressure field to a `.vtu`.
///
/// `basis` selects whether `pressure` is per non-evaluation surface element
/// (`Constant`) or per mesh node (`Linear`).
pub fn write_surface_pressure_vtu(
    mesh: &Mesh,
    pressure: &[Complex64],
    basis: BasisOrder,
    path: impl AsRef<Path>,
) -> std::io::Result<()> {
    if basis == BasisOrder::Quadratic {
        return write_quadratic_surface_pressure_vtu(mesh, &mesh.elements, pressure, path);
    }
    let mut w = VtuWriter::new(mesh);
    match basis {
        BasisOrder::Constant => {
            let n_cells = mesh
                .elements
                .iter()
                .filter(|e| !e.property.is_evaluation())
                .count();
            if pressure.len() != n_cells {
                return Err(std::io::Error::other(format!(
                    "write_surface_pressure_vtu: pressure DOFs ({}) != surface cells ({n_cells})",
                    pressure.len()
                )));
            }
            let mag: Vec<f64> = pressure.iter().map(|v| v.norm()).collect();
            let real: Vec<f64> = pressure.iter().map(|v| v.re).collect();
            let imag: Vec<f64> = pressure.iter().map(|v| v.im).collect();
            let spl: Vec<f64> = pressure.iter().map(|&v| pressure_to_spl(v)).collect();
            w.add_cell_scalar("Pressure_magnitude", &mag);
            w.add_cell_scalar("Pressure_real", &real);
            w.add_cell_scalar("Pressure_imag", &imag);
            w.add_cell_scalar("Pressure_SPL_dB", &spl);
        }
        BasisOrder::Linear => {
            let n_nodes = mesh.nodes.len();
            if pressure.len() != n_nodes {
                return Err(std::io::Error::other(format!(
                    "write_surface_pressure_vtu: pressure DOFs ({}) != mesh nodes ({n_nodes})",
                    pressure.len()
                )));
            }
            w.add_point_scalar_complex("Pressure", pressure);
            let spl: Vec<f64> = pressure.iter().map(|&v| pressure_to_spl(v)).collect();
            w.add_point_scalar("Pressure_SPL_dB", &spl);
        }
        BasisOrder::Quadratic => unreachable!("quadratic export handled above"),
    }
    w.write(path)
}

#[derive(Debug, Default)]
struct QuadraticVisualizationMesh {
    nodes: Vec<[f64; 3]>,
    cells: Vec<(CellKind, Vec<u64>)>,
}

impl MeshView for QuadraticVisualizationMesh {
    fn num_nodes(&self) -> usize {
        self.nodes.len()
    }

    fn num_cells(&self) -> usize {
        self.cells.len()
    }

    fn nodes(&self) -> Box<dyn Iterator<Item = [f64; 3]> + '_> {
        Box::new(self.nodes.iter().copied())
    }

    fn cells(&self) -> Box<dyn Iterator<Item = (CellKind, Vec<u64>)> + '_> {
        Box::new(self.cells.iter().cloned())
    }
}

fn midpoint(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        0.5 * (a[0] + b[0]),
        0.5 * (a[1] + b[1]),
        0.5 * (a[2] + b[2]),
    ]
}

fn element_node(mesh: &Mesh, element: &Element, local: usize) -> std::io::Result<[f64; 3]> {
    let node = *element.connectivity.get(local).ok_or_else(|| {
        std::io::Error::other(format!(
            "quadratic VTU export: element is missing connectivity slot {local}"
        ))
    })?;
    mesh.nodes.get(node).copied().ok_or_else(|| {
        std::io::Error::other(format!(
            "quadratic VTU export: element connectivity node {node} is outside mesh node range"
        ))
    })
}

fn element_pressure_values(
    element: &Element,
    pressure: &[Complex64],
) -> std::io::Result<Vec<Complex64>> {
    let expected = element
        .element_type
        .num_dofs_for_basis(BasisOrder::Quadratic);
    if element.dof_addresses.len() != expected {
        return Err(std::io::Error::other(format!(
            "quadratic VTU export: {:?} element has {} DOFs, expected {expected}",
            element.element_type,
            element.dof_addresses.len()
        )));
    }
    element
        .dof_addresses
        .iter()
        .map(|&dof| {
            pressure.get(dof).copied().ok_or_else(|| {
                std::io::Error::other(format!(
                    "quadratic VTU export: element DOF {dof} is outside pressure vector of length {}",
                    pressure.len()
                ))
            })
        })
        .collect()
}

fn append_quadratic_element(
    output: &mut QuadraticVisualizationMesh,
    field: &mut Vec<Complex64>,
    mesh: &Mesh,
    element: &Element,
    pressure: &[Complex64],
) -> std::io::Result<()> {
    if element.property.is_evaluation() {
        return Ok(());
    }
    let values = element_pressure_values(element, pressure)?;
    let start = output.nodes.len() as u64;
    match element.element_type {
        ElementType::Tri3 => {
            let v0 = element_node(mesh, element, 0)?;
            let v1 = element_node(mesh, element, 1)?;
            let v2 = element_node(mesh, element, 2)?;
            output.nodes.extend([
                v0,
                v1,
                v2,
                midpoint(v0, v1),
                midpoint(v1, v2),
                midpoint(v2, v0),
            ]);
            field.extend(values);
            output.cells.extend([
                (CellKind::Triangle, vec![start, start + 3, start + 5]),
                (CellKind::Triangle, vec![start + 3, start + 1, start + 4]),
                (CellKind::Triangle, vec![start + 5, start + 4, start + 2]),
                (CellKind::Triangle, vec![start + 3, start + 4, start + 5]),
            ]);
        }
        ElementType::Quad4 => {
            let v0 = element_node(mesh, element, 0)?;
            let v1 = element_node(mesh, element, 1)?;
            let v2 = element_node(mesh, element, 2)?;
            let v3 = element_node(mesh, element, 3)?;
            let center = [
                0.25 * (v0[0] + v1[0] + v2[0] + v3[0]),
                0.25 * (v0[1] + v1[1] + v2[1] + v3[1]),
                0.25 * (v0[2] + v1[2] + v2[2] + v3[2]),
            ];
            output.nodes.extend([
                v0,
                v1,
                v2,
                v3,
                midpoint(v0, v1),
                midpoint(v1, v2),
                midpoint(v2, v3),
                midpoint(v3, v0),
                center,
            ]);
            field.extend(values);
            output.cells.extend([
                (CellKind::Quad, vec![start, start + 4, start + 8, start + 7]),
                (
                    CellKind::Quad,
                    vec![start + 4, start + 1, start + 5, start + 8],
                ),
                (
                    CellKind::Quad,
                    vec![start + 8, start + 5, start + 2, start + 6],
                ),
                (
                    CellKind::Quad,
                    vec![start + 7, start + 8, start + 6, start + 3],
                ),
            ]);
        }
    }
    Ok(())
}

/// Write a quadratic surface solution as a refined linear-cell VTU.
///
/// Each Tri6 panel is split into four triangles and each Quad9 panel into four
/// quads. The generated points carry the corresponding quadratic nodal values,
/// including edge-midpoint and Quad4 interior coefficients, so no solved DOF is
/// silently discarded by the export. Evaluation elements are omitted just as
/// they are for the other surface-export paths.
pub(crate) fn write_quadratic_surface_pressure_vtu(
    mesh: &Mesh,
    elements: &[Element],
    pressure: &[Complex64],
    path: impl AsRef<Path>,
) -> std::io::Result<()> {
    let (visualization, field) = build_quadratic_visualization(mesh, elements, pressure)?;
    let mut writer = VtuWriter::new(&visualization);
    writer.add_point_scalar_complex("Pressure", &field);
    let spl: Vec<f64> = field.iter().map(|&value| pressure_to_spl(value)).collect();
    writer.add_point_scalar("Pressure_SPL_dB", &spl);
    writer.write(path)
}

fn build_quadratic_visualization(
    mesh: &Mesh,
    elements: &[Element],
    pressure: &[Complex64],
) -> std::io::Result<(QuadraticVisualizationMesh, Vec<Complex64>)> {
    let mut visualization = QuadraticVisualizationMesh::default();
    let mut field = Vec::new();
    for element in elements {
        append_quadratic_element(&mut visualization, &mut field, mesh, element, pressure)?;
    }
    Ok((visualization, field))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::types::{BoundaryCondition, ElementProperty};

    fn element(
        element_type: ElementType,
        connectivity: Vec<usize>,
        dof_addresses: Vec<usize>,
    ) -> Element {
        Element {
            node_normals: vec![[0.0, 0.0, 1.0]; element_type.num_nodes()],
            connectivity,
            element_type,
            property: ElementProperty::Surface,
            normal: [0.0, 0.0, 1.0],
            center: [0.0, 0.0, 0.0],
            area: 1.0,
            boundary_condition: BoundaryCondition::Velocity(vec![Complex64::default()]),
            group: 0,
            dof_addresses,
        }
    }

    fn mesh(nodes: Vec<[f64; 3]>, elements: Vec<Element>) -> Mesh {
        Mesh {
            num_boundary_nodes: nodes.len(),
            num_boundary_elements: elements.len(),
            nodes,
            elements,
            external_node_numbers: Vec::new(),
            external_element_numbers: Vec::new(),
            num_evaluation_nodes: 0,
            num_evaluation_elements: 0,
            symmetry_planes: [0; 3],
            symmetry_coordinates: [0.0; 3],
            num_reflections: 0,
            node_to_elements: Vec::new(),
        }
    }

    #[test]
    fn quadratic_tri_export_preserves_all_six_values_on_four_cells() {
        let mesh = mesh(
            vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            vec![element(ElementType::Tri3, vec![0, 1, 2], (0..6).collect())],
        );
        let pressure: Vec<_> = (0..6)
            .map(|value| Complex64::new(value as f64, 0.0))
            .collect();
        let (visualization, field) =
            build_quadratic_visualization(&mesh, &mesh.elements, &pressure)
                .expect("valid quadratic triangle");
        assert_eq!(visualization.nodes.len(), 6);
        assert_eq!(visualization.cells.len(), 4);
        assert_eq!(field, pressure);
        assert_eq!(visualization.cells[0].1, vec![0, 3, 5]);
    }

    #[test]
    fn quadratic_quad_export_preserves_center_value_on_four_cells() {
        let mesh = mesh(
            vec![
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            vec![element(
                ElementType::Quad4,
                vec![0, 1, 2, 3],
                (0..9).collect(),
            )],
        );
        let pressure: Vec<_> = (0..9)
            .map(|value| Complex64::new(value as f64, 0.0))
            .collect();
        let (visualization, field) =
            build_quadratic_visualization(&mesh, &mesh.elements, &pressure)
                .expect("valid quadratic quadrilateral");
        assert_eq!(visualization.nodes.len(), 9);
        assert_eq!(visualization.cells.len(), 4);
        assert_eq!(field[8], Complex64::new(8.0, 0.0));
        assert_eq!(visualization.cells[3].1, vec![7, 8, 6, 3]);
    }

    #[test]
    fn quadratic_export_rejects_wrong_dof_count() {
        let mesh = mesh(
            vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            vec![element(ElementType::Tri3, vec![0, 1, 2], vec![0; 5])],
        );
        let error =
            build_quadratic_visualization(&mesh, &mesh.elements, &[Complex64::default(); 5])
                .expect_err("wrong quadratic DOF count must fail");
        assert!(error.to_string().contains("expected 6"));
    }
}

/// Write only the surface mesh (no field data) to a `.vtu`.
pub fn write_mesh_vtu(mesh: &Mesh, path: impl AsRef<Path>) -> std::io::Result<()> {
    VtuWriter::new(mesh).write(path)
}
