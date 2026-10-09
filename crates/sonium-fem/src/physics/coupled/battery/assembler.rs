//! Battery particle and macro-scale operator assembly.

use crate::assembly::TripletMatrix;
use crate::assembly::prebaked::PrebakedAssembler;
use crate::basis::PolynomialDegree;
use crate::mesh::Mesh;
use crate::quadrature;
use crate::spaces::H1Space;

use super::{
    BatteryCapacityForm, BatteryMaterial, BatteryRegionKind, BatteryScalarDiffusionForm,
    SphericalParticleDiffusionForm, SphericalParticleMassForm,
};

/// Particle components: spherical diffusion and spherical storage.
pub type BatteryParticleAssembler = PrebakedAssembler<2>;

pub fn build_battery_particle_assembler(
    radial_mesh: &Mesh,
    degree: PolynomialDegree,
    material: BatteryMaterial,
) -> BatteryParticleAssembler {
    let space = H1Space::new(degree);
    let stiffness_order = quadrature::required_order_for_stiffness(degree.degree());
    let mass_order = quadrature::required_order_for_mass(degree.degree());
    let diffusion = crate::assembly::assemble_bilinear(
        &SphericalParticleDiffusionForm {
            solid_diffusivity: material.solid_diffusivity,
        },
        &space,
        &space,
        radial_mesh,
        stiffness_order,
    );
    let mass = crate::assembly::assemble_bilinear(
        &SphericalParticleMassForm,
        &space,
        &space,
        radial_mesh,
        mass_order,
    );
    PrebakedAssembler::from_triplets([diffusion, mass], &[])
}

pub struct BatteryMacroOperators {
    pub electrolyte_diffusion: TripletMatrix<f64>,
    pub electrolyte_conduction: TripletMatrix<f64>,
    pub solid_conduction: TripletMatrix<f64>,
    pub electrolyte_capacity: TripletMatrix<f64>,
}

pub fn assemble_battery_macro_operators(
    mesh: &Mesh,
    degree: PolynomialDegree,
    material: BatteryMaterial,
) -> BatteryMacroOperators {
    let space = H1Space::new(degree);
    let stiffness_order = quadrature::required_order_for_stiffness(degree.degree());
    let mass_order = quadrature::required_order_for_mass(degree.degree());
    let scalar = |coefficient| {
        crate::assembly::assemble_bilinear(
            &BatteryScalarDiffusionForm { coefficient },
            &space,
            &space,
            mesh,
            stiffness_order,
        )
    };
    let solid_conductivity = if material.region == BatteryRegionKind::Separator {
        0.0
    } else {
        material.solid_conductivity
    };
    BatteryMacroOperators {
        electrolyte_diffusion: scalar(material.effective_electrolyte_diffusivity()),
        electrolyte_conduction: scalar(material.effective_electrolyte_conductivity()),
        solid_conduction: scalar(solid_conductivity),
        electrolyte_capacity: crate::assembly::assemble_bilinear(
            &BatteryCapacityForm {
                coefficient: material.porosity,
            },
            &space,
            &space,
            mesh,
            mass_order,
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::unit_square_triangles;

    fn material() -> BatteryMaterial {
        BatteryMaterial {
            region: BatteryRegionKind::NegativeElectrode,
            porosity: 0.3,
            bruggeman_exponent: 1.5,
            electrolyte_diffusivity: 1e-10,
            electrolyte_conductivity: 1.0,
            solid_conductivity: 100.0,
            transference_number: 0.4,
            particle_radius: 5e-6,
            solid_diffusivity: 1e-14,
            maximum_solid_concentration: 30_000.0,
            specific_surface_area: 3e5,
            exchange_current_density: 2.0,
            charge_transfer_coefficient: 0.5,
            density: 2000.0,
            specific_heat: 1000.0,
            thermal_conductivity: 2.0,
        }
    }

    #[test]
    fn macro_operators_conserve_constant_concentration_and_potential() {
        let mesh = unit_square_triangles(2);
        let operators = assemble_battery_macro_operators(&mesh, PolynomialDegree::P1, material());
        for operator in [
            operators.electrolyte_diffusion,
            operators.electrolyte_conduction,
            operators.solid_conduction,
        ] {
            let csr = operator.to_csr();
            for row in 0..csr.num_rows {
                let sum: f64 = csr.row_entries(row).map(|(_, value)| value).sum();
                assert!(sum.abs() < 1e-12);
            }
        }
    }
}
