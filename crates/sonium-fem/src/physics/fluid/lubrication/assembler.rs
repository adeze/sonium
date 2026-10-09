//! Generic assembly helpers for lubrication operators.

use crate::assembly::TripletMatrix;
use crate::basis::PolynomialDegree;
use crate::mesh::Mesh;
use crate::quadrature;
use crate::spaces::H1Space;

use super::{Lubricant, ReynoldsPressureForm, ReynoldsSqueezeSourceForm, ReynoldsWedgeSourceForm};

pub fn assemble_reynolds_pressure<H>(
    mesh: &Mesh,
    degree: PolynomialDegree,
    lubricant: Lubricant,
    film_thickness: H,
) -> TripletMatrix<f64>
where
    H: Fn(f64, f64, f64) -> f64 + Send + Sync,
{
    let space = H1Space::new(degree);
    let order = quadrature::required_order_for_stiffness(degree.degree());
    crate::assembly::assemble_bilinear(
        &ReynoldsPressureForm {
            lubricant,
            film_thickness,
        },
        &space,
        &space,
        mesh,
        order,
    )
}

pub fn assemble_reynolds_wedge_rhs<H, U>(
    mesh: &Mesh,
    degree: PolynomialDegree,
    film_thickness: H,
    surface_velocity_sum: U,
) -> Vec<f64>
where
    H: Fn(f64, f64, f64) -> f64 + Send + Sync,
    U: Fn(f64, f64, f64) -> [f64; 3] + Send + Sync,
{
    let space = H1Space::new(degree);
    let order = quadrature::required_order_for_stiffness(degree.degree());
    crate::assembly::assemble_linear(
        &ReynoldsWedgeSourceForm {
            film_thickness,
            surface_velocity_sum,
        },
        &space,
        mesh,
        order,
    )
}

pub fn assemble_reynolds_squeeze_rhs<R>(
    mesh: &Mesh,
    degree: PolynomialDegree,
    film_rate: R,
) -> Vec<f64>
where
    R: Fn(f64, f64, f64) -> f64 + Send + Sync,
{
    let space = H1Space::new(degree);
    let order = quadrature::required_order_for_mass(degree.degree());
    crate::assembly::assemble_linear(
        &ReynoldsSqueezeSourceForm { film_rate },
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
    fn constant_film_assembles_conservative_operator() {
        let mesh = unit_square_triangles(3);
        let lubricant = Lubricant {
            dynamic_viscosity: 0.1,
            density: 850.0,
            ambient_pressure: 0.0,
            minimum_film_thickness: 1e-9,
        };
        let matrix =
            assemble_reynolds_pressure(&mesh, PolynomialDegree::P1, lubricant, |_, _, _| 1e-3)
                .to_csr();
        for row in 0..matrix.num_rows {
            let sum: f64 = matrix.row_entries(row).map(|(_, value)| value).sum();
            assert!(
                sum.abs() < 1e-20,
                "constant pressure must be in the Reynolds null space"
            );
        }
    }
}
