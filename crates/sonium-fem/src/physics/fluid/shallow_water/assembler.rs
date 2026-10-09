//! Shallow-water assembler helpers.

use std::collections::HashMap;

use crate::assembly::TripletMatrix;
use crate::assembly::prebaked::PrebakedAssembler;
use crate::basis::PolynomialDegree;
use crate::mesh::Mesh;
use crate::quadrature;
use crate::spaces::H1Space;
use sonium_solvers::CsrMatrix;

use super::{
    BedSlopeSourceForm, FreeSurfaceMassForm, ManningFrictionForm, ShallowWaterGravityWaveForm,
    ShallowWaterMaterial,
};

/// Components:
/// 0 = gravity-wave stiffness, 1 = free-surface mass, 2 = Manning friction.
pub type ShallowWaterAssembler = PrebakedAssembler<3>;

/// Build shallow-water components without boundary contributions.
pub fn build_shallow_water_assembler(
    mesh: &Mesh,
    degree: PolynomialDegree,
    material: ShallowWaterMaterial,
    velocity_magnitude: f64,
) -> ShallowWaterAssembler {
    build_shallow_water_assembler_with_boundaries(mesh, degree, &[], material, velocity_magnitude)
}

/// Build shallow-water components with optional boundary mass terms.
pub fn build_shallow_water_assembler_with_boundaries(
    mesh: &Mesh,
    degree: PolynomialDegree,
    boundary_tags: &[i32],
    material: ShallowWaterMaterial,
    velocity_magnitude: f64,
) -> ShallowWaterAssembler {
    let space = H1Space::new(degree);
    let quad_stiffness = quadrature::required_order_for_stiffness(degree.degree());
    let quad_mass = quadrature::required_order_for_mass(degree.degree());

    let wave = crate::assembly::assemble_bilinear(
        &ShallowWaterGravityWaveForm { material },
        &space,
        &space,
        mesh,
        quad_stiffness,
    );
    let mass =
        crate::assembly::assemble_bilinear(&FreeSurfaceMassForm, &space, &space, mesh, quad_mass);
    let friction = crate::assembly::assemble_bilinear(
        &ManningFrictionForm::new(&material, velocity_magnitude, material.mean_depth),
        &space,
        &space,
        mesh,
        quad_mass,
    );

    let mut boundary_triplets: Vec<(usize, TripletMatrix<f64>)> = Vec::new();
    for &tag in boundary_tags {
        let b_mass = crate::assembly::assemble_boundary_mass_generic(mesh, tag);
        if b_mass.nnz() > 0 {
            boundary_triplets.push((tag as usize, b_mass));
        }
    }

    PrebakedAssembler::from_triplets([wave, mass, friction], &boundary_triplets)
}

/// Assemble bed-slope RHS forcing.
pub fn assemble_bed_slope_rhs(
    mesh: &Mesh,
    degree: PolynomialDegree,
    material: ShallowWaterMaterial,
    bed_slope: [f64; 3],
    flow_direction: Option<[f64; 3]>,
) -> Vec<f64> {
    let space = H1Space::new(degree);
    let quad_mass = quadrature::required_order_for_mass(degree.degree());
    let form = if let Some(direction) = flow_direction {
        BedSlopeSourceForm::with_flow_direction(material, bed_slope, direction)
    } else {
        BedSlopeSourceForm::new(material, bed_slope)
    };
    crate::assembly::assemble_linear(&form, &space, mesh, quad_mass)
}

/// Extension methods for shallow-water matrix assembly.
pub trait ShallowWaterAssemblerExt {
    /// Assemble `A = a0 * K_wave + a1 * M + a2 * M_friction`.
    fn assemble_shallow_water_operator(
        &self,
        wave_scale: f64,
        mass_scale: f64,
        friction_scale: f64,
        boundary_coeffs: &HashMap<usize, f64>,
    ) -> CsrMatrix<f64>;
}

impl ShallowWaterAssemblerExt for ShallowWaterAssembler {
    fn assemble_shallow_water_operator(
        &self,
        wave_scale: f64,
        mass_scale: f64,
        friction_scale: f64,
        boundary_coeffs: &HashMap<usize, f64>,
    ) -> CsrMatrix<f64> {
        let mut system = self.assemble_real(&[wave_scale, mass_scale, friction_scale]);
        for (&tag, &coeff) in boundary_coeffs {
            if let Some(boundary_values) = self.boundary_components.get(&tag) {
                for (a, &b) in system.values.iter_mut().zip(boundary_values.iter()) {
                    *a += coeff * b;
                }
            }
        }
        system
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::unit_square_triangles;

    #[test]
    fn build_shallow_water_assembler_smoke() {
        let mesh = unit_square_triangles(2);
        let material = ShallowWaterMaterial {
            gravity: 9.81,
            mean_depth: 1.5,
            manning_n: 0.02,
        };
        let assembler = build_shallow_water_assembler(&mesh, PolynomialDegree::P1, material, 1.0);
        assert!(assembler.nnz() > 0);
        assert_eq!(assembler.num_rows, mesh.num_nodes());

        let rhs = assemble_bed_slope_rhs(
            &mesh,
            PolynomialDegree::P1,
            material,
            [0.01, 0.0, 0.0],
            Some([1.0, 0.0, 0.0]),
        );
        assert_eq!(rhs.len(), mesh.num_nodes());
    }
}
