//! Marker-routed conjugate heat-transfer assembly.

use crate::assembly::prebaked::PrebakedAssembler;
use crate::assembly::region::{RegionAssemblyError, RegionMap, assemble_bilinear_by_region};
use crate::basis::PolynomialDegree;
use crate::mesh::Mesh;
use crate::quadrature;
use crate::spaces::H1Space;

use super::{ChtAdvectionForm, ChtCapacityForm, ChtConductionForm, ConjugateRegion};

/// Components are conduction, capacity, and fluid advection.
pub type ConjugateHeatTransferAssembler = PrebakedAssembler<3>;

pub fn build_conjugate_heat_transfer_assembler(
    mesh: &Mesh,
    degree: PolynomialDegree,
    regions: &RegionMap<ConjugateRegion>,
) -> Result<ConjugateHeatTransferAssembler, RegionAssemblyError> {
    let mut conduction = RegionMap::new();
    let mut capacity = RegionMap::new();
    let mut advection = RegionMap::new();
    for marker in regions.markers() {
        let region = *regions
            .get_explicit(marker)
            .expect("marker iterator only yields explicit entries");
        conduction.insert(marker, ChtConductionForm::from(region));
        capacity.insert(marker, ChtCapacityForm::from(region));
        advection.insert(marker, ChtAdvectionForm::from(region));
    }

    let space = H1Space::new(degree);
    let stiffness_order = quadrature::required_order_for_stiffness(degree.degree());
    let mass_order = quadrature::required_order_for_mass(degree.degree());
    let k = assemble_bilinear_by_region(&conduction, &space, &space, mesh, stiffness_order)?;
    let m = assemble_bilinear_by_region(&capacity, &space, &space, mesh, mass_order)?;
    let a = assemble_bilinear_by_region(&advection, &space, &space, mesh, stiffness_order)?;
    Ok(PrebakedAssembler::from_triplets([k, m, a], &[]))
}

pub trait ConjugateHeatTransferAssemblerExt {
    /// Assemble `K + capacity_scale M + advection_scale A`.
    fn assemble_cht_operator(
        &self,
        capacity_scale: f64,
        advection_scale: f64,
    ) -> sonium_solvers::CsrMatrix<f64>;
}

impl ConjugateHeatTransferAssemblerExt for ConjugateHeatTransferAssembler {
    fn assemble_cht_operator(
        &self,
        capacity_scale: f64,
        advection_scale: f64,
    ) -> sonium_solvers::CsrMatrix<f64> {
        self.assemble_real(&[1.0, capacity_scale, advection_scale])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::unit_square_triangles;

    #[test]
    fn shared_h1_interface_is_conservative_and_temperature_conforming() {
        let mut mesh = unit_square_triangles(1);
        mesh.elements[0].marker = 1;
        mesh.elements[1].marker = 2;
        let mut regions = RegionMap::new();
        regions.insert(1, ConjugateRegion::solid(10.0, 1000.0, 500.0));
        regions.insert(2, ConjugateRegion::fluid(1.0, 1.0, 1000.0, [2.0, 0.0, 0.0]));
        let assembler =
            build_conjugate_heat_transfer_assembler(&mesh, PolynomialDegree::P1, &regions)
                .expect("both material regions are mapped");
        assert_eq!(assembler.num_rows, mesh.num_nodes());
        let conduction = assembler.assemble_real(&[1.0, 0.0, 0.0]);
        for row in 0..conduction.num_rows {
            let sum: f64 = conduction.row_entries(row).map(|(_, value)| value).sum();
            assert!(
                sum.abs() < 1e-12,
                "constant temperature must be an exact null state"
            );
        }
    }
}
