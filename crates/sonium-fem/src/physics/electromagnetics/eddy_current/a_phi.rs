//! Three-dimensional magnetoquasistatic A-phi formulation.
//!
//! With the `e^(+i omega t)` convention,
//! `E = -i omega A - grad(phi)` and `J = sigma E`. The weak system is
//!
//! ```text
//! [ K_nu + i omega M_sigma    G_sigma ] [A  ]   [J_s]
//! [ i omega G_sigma^T         L_sigma ] [phi] = [ 0 ]
//! ```
//!
//! `A` uses lowest-order H(curl) edge elements and `phi` uses H1 nodal
//! elements. A tree-cotree gauge removes the gradient nullspace of the
//! curl-curl operator in non-conducting regions.

use crate::assembly::TripletMatrix;
use crate::basis::PolynomialDegree;
use crate::mesh::{Mesh, topology::MeshTopology};
use crate::physics::electromagnetics::magnetostatics::MU_0;
use crate::quadrature::QuadratureRule;
use crate::spaces::{FunctionSpace, H1Space, HcurlSpace};
use num_complex::Complex64;
use sonium_solvers::sparse::CsrMatrix;
use std::collections::{BTreeMap, HashSet};
use std::f64::consts::PI;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AphiMaterial {
    pub permeability_h_per_m: f64,
    pub conductivity_s_per_m: f64,
}

impl AphiMaterial {
    pub fn new(permeability_h_per_m: f64, conductivity_s_per_m: f64) -> Self {
        Self {
            permeability_h_per_m,
            conductivity_s_per_m,
        }
    }

    pub fn from_relative_permeability(mu_r: f64, conductivity_s_per_m: f64) -> Self {
        Self::new(mu_r * MU_0, conductivity_s_per_m)
    }

    fn validate(self, marker: i32) -> Result<(), AphiError> {
        if !self.permeability_h_per_m.is_finite() || self.permeability_h_per_m <= 0.0 {
            return Err(AphiError::InvalidPermeability { marker });
        }
        if !self.conductivity_s_per_m.is_finite() || self.conductivity_s_per_m < 0.0 {
            return Err(AphiError::InvalidConductivity { marker });
        }
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct AphiAssembly {
    pub matrix: TripletMatrix<Complex64>,
    pub rhs: Vec<Complex64>,
    pub n_a_dofs: usize,
    pub n_phi_dofs: usize,
    pub active_phi_nodes: HashSet<usize>,
    pub frequency_hz: f64,
}

impl AphiAssembly {
    pub fn total_dofs(&self) -> usize {
        self.n_a_dofs + self.n_phi_dofs
    }

    pub fn inactive_phi_dofs(&self) -> Vec<usize> {
        (0..self.n_phi_dofs)
            .filter(|node| !self.active_phi_nodes.contains(node))
            .map(|node| self.n_a_dofs + node)
            .collect()
    }
}

/// Assemble the coupled A-phi system and impressed-current RHS.
pub fn assemble_aphi<F>(
    mesh: &Mesh,
    materials: &BTreeMap<i32, AphiMaterial>,
    frequency_hz: f64,
    quadrature_order: usize,
    source_current_density: F,
) -> Result<AphiAssembly, AphiError>
where
    F: Fn(i32, [f64; 3]) -> [Complex64; 3] + Send + Sync,
{
    if mesh.dimension != 3 {
        return Err(AphiError::RequiresThreeDimensions(mesh.dimension));
    }
    if !frequency_hz.is_finite() || frequency_hz <= 0.0 {
        return Err(AphiError::InvalidFrequency);
    }
    if quadrature_order == 0 {
        return Err(AphiError::InvalidQuadratureOrder);
    }
    for (&marker, &material) in materials {
        material.validate(marker)?;
    }
    let omega = 2.0 * PI * frequency_hz;
    let a_space = HcurlSpace::new();
    let phi_space = H1Space::new(PolynomialDegree::P1);
    let n_a = a_space.num_global_dofs(mesh);
    let n_phi = phi_space.num_global_dofs(mesh);
    let mut matrix = TripletMatrix::new(n_a + n_phi, n_a + n_phi);
    let mut rhs = vec![Complex64::default(); n_a + n_phi];
    let mut active_phi_nodes = HashSet::new();

    for element_index in 0..mesh.num_elements() {
        let element = &mesh.elements[element_index];
        let material = *materials
            .get(&element.marker)
            .ok_or(AphiError::MissingMaterial {
                marker: element.marker,
            })?;
        let conductivity = material.conductivity_s_per_m;
        let inv_mu = 1.0 / material.permeability_h_per_m;
        let a_dofs = a_space.local_to_global(mesh, element_index);
        let phi_dofs = phi_space.local_to_global(mesh, element_index);
        if conductivity > 0.0 {
            active_phi_nodes.extend(phi_dofs.iter().copied());
        }
        let quadrature = QuadratureRule::new(element.element_type, quadrature_order);
        for point in quadrature.iter() {
            let a_tab =
                a_space.tabulate(mesh, element_index, point.xi(), point.eta(), point.zeta());
            let phi_tab =
                phi_space.tabulate(mesh, element_index, point.xi(), point.eta(), point.zeta());
            let weight = point.weight * a_tab.det_j;
            let source = source_current_density(element.marker, a_tab.physical_coords);

            for (local_i, &global_i) in a_dofs.iter().enumerate() {
                let test_value = hcurl_value(&a_tab, local_i);
                let test_curl = hcurl_curl(&a_tab, local_i);
                rhs[global_i] += complex_dot_real(source, test_value) * weight;
                for (local_j, &global_j) in a_dofs.iter().enumerate() {
                    let trial_value = hcurl_value(&a_tab, local_j);
                    let trial_curl = hcurl_curl(&a_tab, local_j);
                    let curl_curl = inv_mu * real_dot(test_curl, trial_curl);
                    let conductive = conductivity * real_dot(test_value, trial_value);
                    matrix.push(
                        global_i,
                        global_j,
                        Complex64::new(curl_curl, omega * conductive) * weight,
                    );
                }
                if conductivity > 0.0 {
                    for (local_j, &phi_j) in phi_dofs.iter().enumerate() {
                        let coupling = conductivity
                            * (0..3)
                                .map(|component| {
                                    test_value[component] * phi_tab.grad(local_j, component)
                                })
                                .sum::<f64>()
                            * weight;
                        matrix.push(global_i, n_a + phi_j, Complex64::new(coupling, 0.0));
                    }
                }
            }

            if conductivity > 0.0 {
                for (local_i, &phi_i) in phi_dofs.iter().enumerate() {
                    let test_gradient = std::array::from_fn(|d| phi_tab.grad(local_i, d));
                    for (local_j, &global_j) in a_dofs.iter().enumerate() {
                        let trial_value = hcurl_value(&a_tab, local_j);
                        let coupling = conductivity * real_dot(test_gradient, trial_value) * weight;
                        matrix.push(n_a + phi_i, global_j, Complex64::new(0.0, omega * coupling));
                    }
                    for (local_j, &phi_j) in phi_dofs.iter().enumerate() {
                        let trial_gradient = std::array::from_fn(|d| phi_tab.grad(local_j, d));
                        let laplacian =
                            conductivity * real_dot(test_gradient, trial_gradient) * weight;
                        matrix.push(n_a + phi_i, n_a + phi_j, Complex64::new(laplacian, 0.0));
                    }
                }
            }
        }
    }
    Ok(AphiAssembly {
        matrix,
        rhs,
        n_a_dofs: n_a,
        n_phi_dofs: n_phi,
        active_phi_nodes,
        frequency_hz,
    })
}

/// Spanning-tree gauge for lowest-order H(curl) edge DOFs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TreeCotreeGauge {
    pub tree_edge_dofs: Vec<usize>,
}

impl TreeCotreeGauge {
    pub fn from_mesh(mesh: &Mesh) -> Result<Self, AphiError> {
        Self::from_mesh_excluding(mesh, &HashSet::new())
    }

    /// Build a spanning tree while avoiding edge DOFs already used as cyclic
    /// slaves. This prevents a gauge equation from competing with an MPC.
    pub fn from_mesh_excluding(
        mesh: &Mesh,
        excluded_edges: &HashSet<usize>,
    ) -> Result<Self, AphiError> {
        let topology = MeshTopology::build(mesh);
        let mut union_find = UnionFind::new(mesh.num_nodes());
        let mut tree_edge_dofs = Vec::with_capacity(mesh.num_nodes().saturating_sub(1));
        for (edge, &(a, b)) in topology.edges.iter().enumerate() {
            if excluded_edges.contains(&edge) {
                continue;
            }
            if union_find.union(a, b) {
                tree_edge_dofs.push(edge);
            }
        }
        if mesh.num_nodes() > 0 && tree_edge_dofs.len() + 1 != mesh.num_nodes() {
            return Err(AphiError::GaugeGraphDisconnected {
                selected_edges: tree_edge_dofs.len(),
                required_edges: mesh.num_nodes() - 1,
            });
        }
        Ok(Self { tree_edge_dofs })
    }
}

/// Apply homogeneous essential constraints to a complex triplet system.
pub fn apply_zero_constraints(
    matrix: &mut TripletMatrix<Complex64>,
    rhs: &mut [Complex64],
    dofs: &[usize],
) -> Result<(), AphiError> {
    let constrained: HashSet<_> = dofs.iter().copied().collect();
    if constrained.iter().any(|&dof| dof >= rhs.len()) {
        return Err(AphiError::ConstraintOutOfRange);
    }
    for index in 0..matrix.nnz() {
        if constrained.contains(&matrix.rows[index]) || constrained.contains(&matrix.cols[index]) {
            matrix.values[index] = Complex64::default();
        }
    }
    for dof in constrained {
        matrix.push(dof, dof, Complex64::new(1.0, 0.0));
        rhs[dof] = Complex64::default();
    }
    Ok(())
}

/// Apply homogeneous constraints to an assembled CSR matrix, inserting
/// diagonal entries for structurally empty scalar-potential rows when needed.
pub fn apply_zero_constraints_csr(
    matrix: &CsrMatrix<Complex64>,
    rhs: &mut [Complex64],
    dofs: &[usize],
) -> Result<CsrMatrix<Complex64>, AphiError> {
    let constrained: HashSet<_> = dofs.iter().copied().collect();
    if rhs.len() != matrix.num_rows
        || matrix.num_rows != matrix.num_cols
        || constrained.iter().any(|&dof| dof >= rhs.len())
    {
        return Err(AphiError::ConstraintOutOfRange);
    }
    let mut triplets = Vec::new();
    for row in 0..matrix.num_rows {
        for (column, value) in matrix.row_entries(row) {
            if !constrained.contains(&row) && !constrained.contains(&column) {
                triplets.push((row, column, value));
            }
        }
    }
    for &dof in &constrained {
        triplets.push((dof, dof, Complex64::new(1.0, 0.0)));
        rhs[dof] = Complex64::default();
    }
    Ok(CsrMatrix::from_triplets(
        matrix.num_rows,
        matrix.num_cols,
        triplets,
    ))
}

#[derive(Debug, Clone)]
pub struct AphiCellField {
    pub marker: i32,
    pub position_m: [f64; 3],
    pub vector_potential_t_m: [Complex64; 3],
    pub flux_density_t: [Complex64; 3],
    pub electric_field_v_per_m: [Complex64; 3],
    pub current_density_a_per_m2: [Complex64; 3],
    pub eddy_loss_density_w_per_m3: f64,
    pub volume_m3: f64,
}

#[derive(Debug, Clone)]
pub struct AphiFieldSummary {
    pub cells: Vec<AphiCellField>,
    pub eddy_loss_w: f64,
    pub magnetic_energy_j: f64,
    pub source_flux_linkage_wb_turn: Complex64,
}

/// Recover cell-centroid A, B, E, J, loss, energy, and source linkage.
pub fn evaluate_aphi_fields<F>(
    mesh: &Mesh,
    materials: &BTreeMap<i32, AphiMaterial>,
    frequency_hz: f64,
    solution: &[Complex64],
    source_current_density_per_amp: F,
) -> Result<AphiFieldSummary, AphiError>
where
    F: Fn(i32, [f64; 3]) -> [Complex64; 3],
{
    let a_space = HcurlSpace::new();
    let phi_space = H1Space::new(PolynomialDegree::P1);
    let n_a = a_space.num_global_dofs(mesh);
    let n_phi = phi_space.num_global_dofs(mesh);
    if solution.len() != n_a + n_phi {
        return Err(AphiError::SolutionLength {
            actual: solution.len(),
            expected: n_a + n_phi,
        });
    }
    let omega = 2.0 * PI * frequency_hz;
    let mut cells = Vec::with_capacity(mesh.num_elements());
    let mut eddy_loss_w = 0.0;
    let mut magnetic_energy_j = 0.0;
    let mut source_flux_linkage_wb_turn = Complex64::default();
    for element_index in 0..mesh.num_elements() {
        let element = &mesh.elements[element_index];
        let material = *materials
            .get(&element.marker)
            .ok_or(AphiError::MissingMaterial {
                marker: element.marker,
            })?;
        let a_tab = a_space.tabulate(mesh, element_index, 0.25, 0.25, 0.25);
        let phi_tab = phi_space.tabulate(mesh, element_index, 0.25, 0.25, 0.25);
        let a_dofs = a_space.local_to_global(mesh, element_index);
        let phi_dofs = phi_space.local_to_global(mesh, element_index);
        let (vector_potential, flux_density, electric_field, current_density) = recover_aphi_point(
            &a_tab,
            &phi_tab,
            &a_dofs,
            &phi_dofs,
            solution,
            n_a,
            omega,
            material.conductivity_s_per_m,
        );
        let volume = tetrahedron_volume(mesh, element.vertices())?;
        let quadrature = QuadratureRule::new(element.element_type, 2);
        let mut cell_loss = 0.0;
        for point in quadrature.iter() {
            let quadrature_a =
                a_space.tabulate(mesh, element_index, point.xi(), point.eta(), point.zeta());
            let quadrature_phi =
                phi_space.tabulate(mesh, element_index, point.xi(), point.eta(), point.zeta());
            let (a, b, _, j) = recover_aphi_point(
                &quadrature_a,
                &quadrature_phi,
                &a_dofs,
                &phi_dofs,
                solution,
                n_a,
                omega,
                material.conductivity_s_per_m,
            );
            let weight = quadrature_a.det_j * point.weight;
            let loss_density = if material.conductivity_s_per_m > 0.0 {
                j.iter().map(|value| value.norm_sqr()).sum::<f64>() / material.conductivity_s_per_m
            } else {
                0.0
            };
            let magnetic_density = 0.5 * b.iter().map(|value| value.norm_sqr()).sum::<f64>()
                / material.permeability_h_per_m;
            let source =
                source_current_density_per_amp(element.marker, quadrature_a.physical_coords);
            source_flux_linkage_wb_turn += complex_dot_conjugate(source, a) * weight;
            cell_loss += loss_density * weight;
            magnetic_energy_j += magnetic_density * weight;
        }
        let eddy_loss_density = cell_loss / volume;
        eddy_loss_w += cell_loss;
        cells.push(AphiCellField {
            marker: element.marker,
            position_m: a_tab.physical_coords,
            vector_potential_t_m: vector_potential,
            flux_density_t: flux_density,
            electric_field_v_per_m: electric_field,
            current_density_a_per_m2: current_density,
            eddy_loss_density_w_per_m3: eddy_loss_density,
            volume_m3: volume,
        });
    }
    Ok(AphiFieldSummary {
        cells,
        eddy_loss_w,
        magnetic_energy_j,
        source_flux_linkage_wb_turn,
    })
}

#[allow(clippy::too_many_arguments)]
fn recover_aphi_point(
    a_tab: &crate::spaces::BasisTabulation,
    phi_tab: &crate::spaces::BasisTabulation,
    a_dofs: &[usize],
    phi_dofs: &[usize],
    solution: &[Complex64],
    n_a: usize,
    omega: f64,
    conductivity: f64,
) -> (
    [Complex64; 3],
    [Complex64; 3],
    [Complex64; 3],
    [Complex64; 3],
) {
    let mut vector_potential = [Complex64::default(); 3];
    let mut flux_density = [Complex64::default(); 3];
    for (local, &dof) in a_dofs.iter().enumerate() {
        for component in 0..3 {
            vector_potential[component] += solution[dof] * a_tab.vector_value(local, component);
            flux_density[component] += solution[dof] * a_tab.derivatives[local * 3 + component];
        }
    }
    let mut grad_phi = [Complex64::default(); 3];
    for (local, &node) in phi_dofs.iter().enumerate() {
        for (component, value) in grad_phi.iter_mut().enumerate() {
            *value += solution[n_a + node] * phi_tab.grad(local, component);
        }
    }
    let electric_field = std::array::from_fn(|component| {
        Complex64::new(0.0, -omega) * vector_potential[component] - grad_phi[component]
    });
    let current_density = electric_field.map(|value| conductivity * value);
    (
        vector_potential,
        flux_density,
        electric_field,
        current_density,
    )
}

fn hcurl_value(tab: &crate::spaces::BasisTabulation, local: usize) -> [f64; 3] {
    std::array::from_fn(|component| tab.vector_value(local, component))
}

fn hcurl_curl(tab: &crate::spaces::BasisTabulation, local: usize) -> [f64; 3] {
    std::array::from_fn(|component| tab.derivatives[local * 3 + component])
}

fn real_dot(left: [f64; 3], right: [f64; 3]) -> f64 {
    (0..3)
        .map(|component| left[component] * right[component])
        .sum()
}

fn complex_dot_real(left: [Complex64; 3], right: [f64; 3]) -> Complex64 {
    (0..3)
        .map(|component| left[component] * right[component])
        .sum()
}

fn complex_dot_conjugate(left: [Complex64; 3], right: [Complex64; 3]) -> Complex64 {
    (0..3)
        .map(|component| left[component].conj() * right[component])
        .sum()
}

fn tetrahedron_volume(mesh: &Mesh, nodes: &[usize]) -> Result<f64, AphiError> {
    if nodes.len() != 4 {
        return Err(AphiError::RequiresTetrahedra);
    }
    let a = &mesh.nodes[nodes[0]];
    let b = &mesh.nodes[nodes[1]];
    let c = &mesh.nodes[nodes[2]];
    let d = &mesh.nodes[nodes[3]];
    let ab = [b.x - a.x, b.y - a.y, b.z - a.z];
    let ac = [c.x - a.x, c.y - a.y, c.z - a.z];
    let ad = [d.x - a.x, d.y - a.y, d.z - a.z];
    let cross = [
        ac[1] * ad[2] - ac[2] * ad[1],
        ac[2] * ad[0] - ac[0] * ad[2],
        ac[0] * ad[1] - ac[1] * ad[0],
    ];
    Ok(real_dot(ab, cross).abs() / 6.0)
}

#[derive(Debug)]
struct UnionFind {
    parent: Vec<usize>,
    rank: Vec<u8>,
}

impl UnionFind {
    fn new(size: usize) -> Self {
        Self {
            parent: (0..size).collect(),
            rank: vec![0; size],
        }
    }

    fn root(&mut self, value: usize) -> usize {
        if self.parent[value] != value {
            self.parent[value] = self.root(self.parent[value]);
        }
        self.parent[value]
    }

    fn union(&mut self, left: usize, right: usize) -> bool {
        let mut left_root = self.root(left);
        let mut right_root = self.root(right);
        if left_root == right_root {
            return false;
        }
        if self.rank[left_root] < self.rank[right_root] {
            std::mem::swap(&mut left_root, &mut right_root);
        }
        self.parent[right_root] = left_root;
        if self.rank[left_root] == self.rank[right_root] {
            self.rank[left_root] += 1;
        }
        true
    }
}

#[derive(Debug, thiserror::Error)]
pub enum AphiError {
    #[error("A-phi assembly requires a 3D mesh, got dimension {0}")]
    RequiresThreeDimensions(usize),
    #[error("A-phi field recovery currently requires tetrahedra")]
    RequiresTetrahedra,
    #[error("frequency must be finite and positive")]
    InvalidFrequency,
    #[error("quadrature order must be positive")]
    InvalidQuadratureOrder,
    #[error("mesh region {marker} has no A-phi material")]
    MissingMaterial { marker: i32 },
    #[error("region {marker} has invalid permeability")]
    InvalidPermeability { marker: i32 },
    #[error("region {marker} has invalid conductivity")]
    InvalidConductivity { marker: i32 },
    #[error("tree gauge graph is disconnected: selected {selected_edges}, need {required_edges}")]
    GaugeGraphDisconnected {
        selected_edges: usize,
        required_edges: usize,
    },
    #[error("essential constraint DOF is outside the system")]
    ConstraintOutOfRange,
    #[error("solution length {actual} does not match expected {expected}")]
    SolutionLength { actual: usize, expected: usize },
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::unit_cube_tetrahedra;

    fn uniform_materials(conductivity: f64) -> BTreeMap<i32, AphiMaterial> {
        BTreeMap::from([(
            0,
            AphiMaterial::from_relative_permeability(1.0, conductivity),
        )])
    }

    #[test]
    fn a_phi_blocks_have_expected_dimensions_and_frequency_scaling() {
        let mesh = unit_cube_tetrahedra(1);
        let source = |_: i32, _: [f64; 3]| {
            [
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(1.0, 0.0),
            ]
        };
        let low = assemble_aphi(&mesh, &uniform_materials(2.0), 10.0, 2, source).unwrap();
        let high = assemble_aphi(&mesh, &uniform_materials(2.0), 20.0, 2, source).unwrap();
        assert_eq!(low.matrix.num_rows, low.total_dofs());
        assert_eq!(low.rhs.len(), low.total_dofs());
        assert_eq!(low.active_phi_nodes.len(), mesh.num_nodes());
        let low_imag: f64 = low.matrix.values.iter().map(|value| value.im.abs()).sum();
        let high_imag: f64 = high.matrix.values.iter().map(|value| value.im.abs()).sum();
        assert!((high_imag / low_imag - 2.0).abs() < 1.0e-12);
    }

    #[test]
    fn tree_gauge_selects_one_less_edge_than_nodes() {
        let mesh = unit_cube_tetrahedra(2);
        let gauge = TreeCotreeGauge::from_mesh(&mesh).unwrap();
        assert_eq!(gauge.tree_edge_dofs.len() + 1, mesh.num_nodes());
        let unique: HashSet<_> = gauge.tree_edge_dofs.iter().copied().collect();
        assert_eq!(unique.len(), gauge.tree_edge_dofs.len());
    }

    #[test]
    fn nonconducting_regions_leave_scalar_potential_inactive() {
        let mesh = unit_cube_tetrahedra(1);
        let assembly = assemble_aphi(&mesh, &uniform_materials(0.0), 100.0, 2, |_, _| {
            [Complex64::default(); 3]
        })
        .unwrap();
        assert!(assembly.active_phi_nodes.is_empty());
        assert_eq!(assembly.inactive_phi_dofs().len(), mesh.num_nodes());
    }

    #[test]
    fn field_recovery_reports_nonnegative_joule_loss() {
        let mesh = unit_cube_tetrahedra(1);
        let materials = uniform_materials(3.0);
        let n_a = HcurlSpace::new().num_global_dofs(&mesh);
        let mut solution = vec![Complex64::default(); n_a + mesh.num_nodes()];
        solution[0] = Complex64::new(1.0, -0.5);
        let summary = evaluate_aphi_fields(&mesh, &materials, 50.0, &solution, |_, _| {
            [Complex64::new(1.0, 0.0); 3]
        })
        .unwrap();
        assert!(summary.eddy_loss_w >= 0.0);
        assert!(summary.magnetic_energy_j > 0.0);
        assert!(
            summary
                .cells
                .iter()
                .all(|cell| cell.eddy_loss_density_w_per_m3 >= 0.0)
        );
    }
}
