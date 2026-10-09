//! Assembly bridges for staggered DC and induction-heating solves.

use num_complex::Complex64;

use crate::assembly::TripletMatrix;
use crate::basis::PolynomialDegree;
use crate::mesh::Mesh;
use crate::quadrature;
use crate::spaces::H1Space;

use super::{DcCurrentForm, DcJouleHeatForm, ElectrothermalMaterial, HarmonicJouleHeatForm};

pub fn assemble_dc_current_operator<T>(
    mesh: &Mesh,
    degree: PolynomialDegree,
    material: ElectrothermalMaterial,
    temperature: T,
) -> TripletMatrix<f64>
where
    T: Fn(f64, f64, f64) -> f64 + Send + Sync,
{
    let space = H1Space::new(degree);
    let order = quadrature::required_order_for_stiffness(degree.degree());
    crate::assembly::assemble_bilinear(
        &DcCurrentForm {
            material,
            temperature,
        },
        &space,
        &space,
        mesh,
        order,
    )
}

pub fn assemble_dc_joule_heat<T, G>(
    mesh: &Mesh,
    degree: PolynomialDegree,
    material: ElectrothermalMaterial,
    temperature: T,
    potential_gradient: G,
) -> Vec<f64>
where
    T: Fn(f64, f64, f64) -> f64 + Send + Sync,
    G: Fn(f64, f64, f64) -> [f64; 3] + Send + Sync,
{
    let space = H1Space::new(degree);
    let order = quadrature::required_order_for_mass(degree.degree());
    crate::assembly::assemble_linear(
        &DcJouleHeatForm {
            material,
            temperature,
            potential_gradient,
        },
        &space,
        mesh,
        order,
    )
}

pub fn assemble_harmonic_joule_heat<T, E>(
    mesh: &Mesh,
    degree: PolynomialDegree,
    material: ElectrothermalMaterial,
    temperature: T,
    electric_field: E,
) -> Vec<f64>
where
    T: Fn(f64, f64, f64) -> f64 + Send + Sync,
    E: Fn(f64, f64, f64) -> [Complex64; 3] + Send + Sync,
{
    let space = H1Space::new(degree);
    let order = quadrature::required_order_for_mass(degree.degree());
    crate::assembly::assemble_linear(
        &HarmonicJouleHeatForm {
            material,
            temperature,
            electric_field,
        },
        &space,
        mesh,
        order,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::unit_square_triangles;

    #[test]
    fn assembled_dc_operator_conserves_charge() {
        let mesh = unit_square_triangles(2);
        let material = ElectrothermalMaterial {
            reference_electrical_conductivity: 1.0e6,
            resistivity_temperature_coefficient: 0.004,
            reference_temperature: 293.15,
            relative_permeability: 1.0,
            thermal_conductivity: 100.0,
            density: 8000.0,
            specific_heat: 400.0,
        };
        let matrix =
            assemble_dc_current_operator(&mesh, PolynomialDegree::P1, material, |_, _, _| 300.0)
                .to_csr();
        for row in 0..matrix.num_rows {
            let sum: f64 = matrix.row_entries(row).map(|(_, value)| value).sum();
            assert!(
                sum.abs() < 1e-8,
                "constant potential must be the exact current null state"
            );
        }
    }
}
