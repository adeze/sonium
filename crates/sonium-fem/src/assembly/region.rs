//! Cell-marker-aware finite-element assembly.
//!
//! This module routes generic variational forms by [`Mesh`](crate::mesh::Mesh)
//! cell marker. It is deliberately physics-agnostic: materials create forms in
//! their own physics modules, then register those forms in a [`RegionMap`].

use super::generic::{assemble_element_bilinear, assemble_element_linear};
use super::triplet::TripletMatrix;
use crate::forms::{BilinearForm, LinearForm};
use crate::mesh::{Element, Mesh};
use crate::parallelism::{
    ELEMENTS_PER_THREAD, collect_indexed, filter_collect_indexed, merge_element_triplets,
};
use crate::spaces::FunctionSpace;
use std::collections::BTreeMap;

type ElementMatrixTriplets<T> = Vec<(usize, usize, T)>;
type RoutedMatrixContributions<T> = Vec<Result<ElementMatrixTriplets<T>, RegionAssemblyError>>;
type ElementVectorEntries<T> = Vec<(usize, T)>;
type RoutedVectorContributions<T> = Vec<Result<ElementVectorEntries<T>, RegionAssemblyError>>;

/// Marker-to-value routing table with an optional fallback value.
#[derive(Debug, Clone)]
pub struct RegionMap<T> {
    entries: BTreeMap<i32, T>,
    fallback: Option<T>,
}

impl<T> Default for RegionMap<T> {
    fn default() -> Self {
        Self {
            entries: BTreeMap::new(),
            fallback: None,
        }
    }
}

impl<T> RegionMap<T> {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_fallback(fallback: T) -> Self {
        Self {
            entries: BTreeMap::new(),
            fallback: Some(fallback),
        }
    }

    /// Insert or replace the value associated with a cell marker.
    pub fn insert(&mut self, marker: i32, value: T) -> Option<T> {
        self.entries.insert(marker, value)
    }

    pub fn get(&self, marker: i32) -> Option<&T> {
        self.entries.get(&marker).or(self.fallback.as_ref())
    }

    pub fn get_explicit(&self, marker: i32) -> Option<&T> {
        self.entries.get(&marker)
    }

    pub fn contains_marker(&self, marker: i32) -> bool {
        self.entries.contains_key(&marker)
    }

    pub fn markers(&self) -> impl ExactSizeIterator<Item = i32> + '_ {
        self.entries.keys().copied()
    }

    pub fn set_fallback(&mut self, fallback: Option<T>) -> Option<T> {
        std::mem::replace(&mut self.fallback, fallback)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum RegionAssemblyError {
    #[error("cell {element_index} has unmapped region marker {marker}")]
    MissingRegion { element_index: usize, marker: i32 },
}

/// Assemble one bilinear form per cell marker.
pub fn assemble_bilinear_by_region<T, F, S>(
    forms: &RegionMap<F>,
    test_space: &S,
    trial_space: &S,
    mesh: &Mesh,
    quad_order: usize,
) -> Result<TripletMatrix<T>, RegionAssemblyError>
where
    T: Copy + Default + std::ops::AddAssign + Send + Sync,
    F: BilinearForm<T>,
    S: FunctionSpace,
{
    let per_element: Vec<_> =
        collect_indexed(mesh.num_elements(), ELEMENTS_PER_THREAD, |element_index| {
            let marker = mesh.elements[element_index].marker;
            let form = forms
                .get(marker)
                .ok_or(RegionAssemblyError::MissingRegion {
                    element_index,
                    marker,
                })?;
            Ok(assemble_element_bilinear(
                form,
                test_space,
                trial_space,
                mesh,
                element_index,
                quad_order,
            ))
        });

    merge_bilinear_contributions(
        test_space.num_global_dofs(mesh),
        trial_space.num_global_dofs(mesh),
        per_element,
    )
}

/// Assemble a bilinear form only on cells accepted by `select`.
pub fn assemble_bilinear_where<T, F, S, P>(
    form: &F,
    test_space: &S,
    trial_space: &S,
    mesh: &Mesh,
    quad_order: usize,
    select: P,
) -> TripletMatrix<T>
where
    T: Copy + Default + std::ops::AddAssign + Send + Sync,
    F: BilinearForm<T>,
    S: FunctionSpace,
    P: Fn(usize, &Element) -> bool + Send + Sync,
{
    let per_element: Vec<_> =
        filter_collect_indexed(mesh.num_elements(), ELEMENTS_PER_THREAD, |element_index| {
            select(element_index, &mesh.elements[element_index]).then(|| {
                assemble_element_bilinear(
                    form,
                    test_space,
                    trial_space,
                    mesh,
                    element_index,
                    quad_order,
                )
            })
        });

    merge_bilinear_contributions(
        test_space.num_global_dofs(mesh),
        trial_space.num_global_dofs(mesh),
        per_element.into_iter().map(Ok).collect(),
    )
    .expect("selected-cell assembly has no routing errors")
}

/// Assemble one linear form per cell marker.
pub fn assemble_linear_by_region<T, F, S>(
    forms: &RegionMap<F>,
    test_space: &S,
    mesh: &Mesh,
    quad_order: usize,
) -> Result<Vec<T>, RegionAssemblyError>
where
    T: Copy + Default + std::ops::AddAssign + Send + Sync,
    F: LinearForm<T>,
    S: FunctionSpace,
{
    let per_element: Vec<_> =
        collect_indexed(mesh.num_elements(), ELEMENTS_PER_THREAD, |element_index| {
            let marker = mesh.elements[element_index].marker;
            let form = forms
                .get(marker)
                .ok_or(RegionAssemblyError::MissingRegion {
                    element_index,
                    marker,
                })?;
            Ok(assemble_element_linear(
                form,
                test_space,
                mesh,
                element_index,
                quad_order,
            ))
        });

    merge_linear_contributions(test_space.num_global_dofs(mesh), per_element)
}

/// Assemble a linear form only on cells accepted by `select`.
pub fn assemble_linear_where<T, F, S, P>(
    form: &F,
    test_space: &S,
    mesh: &Mesh,
    quad_order: usize,
    select: P,
) -> Vec<T>
where
    T: Copy + Default + std::ops::AddAssign + Send + Sync,
    F: LinearForm<T>,
    S: FunctionSpace,
    P: Fn(usize, &Element) -> bool + Send + Sync,
{
    let per_element: Vec<_> =
        filter_collect_indexed(mesh.num_elements(), ELEMENTS_PER_THREAD, |element_index| {
            select(element_index, &mesh.elements[element_index])
                .then(|| assemble_element_linear(form, test_space, mesh, element_index, quad_order))
        });
    merge_linear_contributions(
        test_space.num_global_dofs(mesh),
        per_element.into_iter().map(Ok).collect(),
    )
    .expect("selected-cell assembly has no routing errors")
}

fn merge_bilinear_contributions<T>(
    num_rows: usize,
    num_cols: usize,
    contributions: RoutedMatrixContributions<T>,
) -> Result<TripletMatrix<T>, RegionAssemblyError>
where
    T: Copy + Default + std::ops::AddAssign + Send + Sync,
{
    let contributions: Vec<_> = contributions.into_iter().collect::<Result<_, _>>()?;
    let total_entries = contributions.iter().map(Vec::len).sum();
    Ok(merge_element_triplets(
        contributions,
        num_rows,
        num_cols,
        total_entries,
    ))
}

fn merge_linear_contributions<T>(
    num_dofs: usize,
    contributions: RoutedVectorContributions<T>,
) -> Result<Vec<T>, RegionAssemblyError>
where
    T: Copy + Default + std::ops::AddAssign,
{
    let mut global = vec![T::default(); num_dofs];
    for element in contributions.into_iter().collect::<Result<Vec<_>, _>>()? {
        for (dof, value) in element {
            global[dof] += value;
        }
    }
    Ok(global)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::basis::PolynomialDegree;
    use crate::forms::{MassForm, SourceForm, WeightedMassForm};
    use crate::mesh::unit_square_triangles;
    use crate::spaces::H1Space;

    #[test]
    fn region_forms_route_distinct_material_coefficients() {
        let mut mesh = unit_square_triangles(1);
        mesh.elements[0].marker = 10;
        mesh.elements[1].marker = 20;
        let space = H1Space::new(PolynomialDegree::P1);
        let mut forms = RegionMap::new();
        forms.insert(10, WeightedMassForm { coefficient: 2.0 });
        forms.insert(20, WeightedMassForm { coefficient: 5.0 });

        let matrix = assemble_bilinear_by_region(&forms, &space, &space, &mesh, 2)
            .expect("both regions mapped")
            .to_csr();
        let integral: f64 = (0..matrix.num_rows)
            .flat_map(|row| matrix.row_entries(row).map(|(_, value)| value))
            .sum();

        assert!((integral - 3.5).abs() < 1.0e-12);
    }

    #[test]
    fn missing_region_reports_marker_and_cell() {
        let mut mesh = unit_square_triangles(1);
        mesh.elements[0].marker = 7;
        mesh.elements[1].marker = 9;
        let space = H1Space::new(PolynomialDegree::P1);
        let mut forms = RegionMap::new();
        forms.insert(7, MassForm);

        let error = assemble_bilinear_by_region(&forms, &space, &space, &mesh, 2)
            .expect_err("marker 9 is not mapped");
        assert_eq!(
            error,
            RegionAssemblyError::MissingRegion {
                element_index: 1,
                marker: 9
            }
        );
    }

    #[test]
    fn selected_cell_assembly_integrates_only_selected_half() {
        let mut mesh = unit_square_triangles(1);
        mesh.elements[0].marker = 1;
        mesh.elements[1].marker = 2;
        let space = H1Space::new(PolynomialDegree::P1);
        let matrix = assemble_bilinear_where(&MassForm, &space, &space, &mesh, 2, |_, element| {
            element.marker == 1
        })
        .to_csr();
        let integral: f64 = (0..matrix.num_rows)
            .flat_map(|row| matrix.row_entries(row).map(|(_, value)| value))
            .sum();
        assert!((integral - 0.5).abs() < 1.0e-12);

        let source = SourceForm::new(|_, _, _| 1.0);
        let rhs =
            assemble_linear_where(&source, &space, &mesh, 2, |_, element| element.marker == 2);
        assert!((rhs.iter().sum::<f64>() - 0.5).abs() < 1.0e-12);
    }
}
