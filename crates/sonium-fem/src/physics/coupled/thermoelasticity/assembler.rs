//! Thermoelastic assembly helpers.

use crate::assembly::TripletMatrix;
use crate::assembly::prebaked::PrebakedAssembler;
use crate::basis::PolynomialDegree;
use crate::mesh::Mesh;
use crate::physics::solid::elasticity::{ElasticityForm, IsotropicMaterial, PlaneAssumption};
use crate::physics::thermal::heat_conduction::{ConductionForm, HeatCapacityForm, ThermalMaterial};
use crate::quadrature;
use crate::spaces::{H1Space, VectorH1Space};
use sonium_solvers::CsrMatrix;

use super::{ThermalStrainLoadForm, ThermoelasticDampingForm, ThermoelasticMaterial};

/// Single component:
/// 0 = thermoelastic damping mass-like term.
///
/// Note: this assembler only covers equal-order scalar components.
/// Mechanical stiffness and thermal-load vectors are assembled through helper
/// functions below because they can live on different spaces.
pub type ThermoelasticAssembler = PrebakedAssembler<1>;

/// Staggered thermoelastic block matrices using the existing scalar/vector H1
/// infrastructure. The blocks can be coupled by callers in monolithic or
/// staggered solves without forcing one global block format on every use case.
pub struct ThermoelasticCoupledBlocks {
    pub mechanical_stiffness: TripletMatrix<f64>,
    pub heat_conduction: TripletMatrix<f64>,
    pub heat_capacity: TripletMatrix<f64>,
    pub damping_coupling: TripletMatrix<f64>,
}

/// Build thermoelastic damping assembler.
pub fn build_thermoelastic_assembler(
    mesh: &Mesh,
    degree: PolynomialDegree,
    coupling_coefficient: f64,
) -> ThermoelasticAssembler {
    let space = H1Space::new(degree);
    let quad_mass = quadrature::required_order_for_mass(degree.degree());
    let damping = crate::assembly::assemble_bilinear(
        &ThermoelasticDampingForm {
            coupling_coefficient,
        },
        &space,
        &space,
        mesh,
        quad_mass,
    );
    PrebakedAssembler::from_triplets([damping], &[])
}

pub fn build_thermoelastic_coupled_blocks(
    mesh: &Mesh,
    degree: PolynomialDegree,
    components: usize,
    elastic_material: IsotropicMaterial,
    plane: Option<PlaneAssumption>,
    thermal_material: ThermalMaterial,
    coupling_coefficient: f64,
) -> ThermoelasticCoupledBlocks {
    let scalar_space = H1Space::new(degree);
    let quad_stiffness = quadrature::required_order_for_stiffness(degree.degree());
    let quad_mass = quadrature::required_order_for_mass(degree.degree());

    let mechanical_stiffness =
        assemble_thermoelastic_stiffness(mesh, degree, components, elastic_material, plane);
    let heat_conduction = crate::assembly::assemble_bilinear(
        &ConductionForm::new(&thermal_material),
        &scalar_space,
        &scalar_space,
        mesh,
        quad_stiffness,
    );
    let heat_capacity = crate::assembly::assemble_bilinear(
        &HeatCapacityForm::new(&thermal_material),
        &scalar_space,
        &scalar_space,
        mesh,
        quad_mass,
    );
    let damping_coupling = crate::assembly::assemble_bilinear(
        &ThermoelasticDampingForm {
            coupling_coefficient,
        },
        &scalar_space,
        &scalar_space,
        mesh,
        quad_mass,
    );

    ThermoelasticCoupledBlocks {
        mechanical_stiffness,
        heat_conduction,
        heat_capacity,
        damping_coupling,
    }
}

/// Assemble elasticity stiffness for the displacement block.
pub fn assemble_thermoelastic_stiffness(
    mesh: &Mesh,
    degree: PolynomialDegree,
    components: usize,
    material: IsotropicMaterial,
    plane: Option<PlaneAssumption>,
) -> TripletMatrix<f64> {
    let space = VectorH1Space::new(degree, components);
    let quad_stiffness = quadrature::required_order_for_stiffness(degree.degree());
    let form = match plane {
        Some(assumption) => ElasticityForm::new_2d(material, assumption),
        None => ElasticityForm::new_3d(material),
    };
    crate::assembly::assemble_bilinear(&form, &space, &space, mesh, quad_stiffness)
}

/// Assemble thermal-strain load vector on a displacement space.
pub fn assemble_thermal_strain_load(
    mesh: &Mesh,
    degree: PolynomialDegree,
    components: usize,
    material: ThermoelasticMaterial,
    temperature: f64,
) -> Vec<f64> {
    let space = VectorH1Space::new(degree, components);
    let quad_stiffness = quadrature::required_order_for_stiffness(degree.degree());
    crate::assembly::assemble_linear(
        &ThermalStrainLoadForm {
            material,
            temperature,
        },
        &space,
        mesh,
        quad_stiffness,
    )
}

/// Extension methods for damping matrix assembly.
pub trait ThermoelasticAssemblerExt {
    /// Assemble scaled damping operator.
    fn assemble_damping_operator(&self, scale: f64) -> CsrMatrix<f64>;
}

impl ThermoelasticAssemblerExt for ThermoelasticAssembler {
    fn assemble_damping_operator(&self, scale: f64) -> CsrMatrix<f64> {
        self.assemble_real(&[scale])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::unit_square_triangles;

    #[test]
    fn build_thermoelastic_assembler_smoke() {
        let mesh = unit_square_triangles(2);
        let assembler = build_thermoelastic_assembler(&mesh, PolynomialDegree::P1, 1.0);
        assert!(assembler.nnz() > 0);
        assert_eq!(assembler.num_rows, mesh.num_nodes());

        let load = assemble_thermal_strain_load(
            &mesh,
            PolynomialDegree::P1,
            2,
            ThermoelasticMaterial {
                young_modulus: 210.0e9,
                poisson_ratio: 0.3,
                thermal_expansion: 12.0e-6,
                reference_temperature: 293.0,
            },
            313.0,
        );
        assert_eq!(load.len(), 2 * mesh.num_nodes());
    }
}
