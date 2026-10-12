//! Solved core–air–core interface benchmark with flux and energy references.

use sonium_fem::assembly::assemble_bilinear;
use sonium_fem::basis::PolynomialDegree;
use sonium_fem::boundary::generic::{FemSystem, try_apply_dirichlet_values};
use sonium_fem::forms::{BilinearForm, LocalMatrix};
use sonium_fem::mesh::{CCoreMeshConfig, CCoreRegion, c_core_mesh_triangles, unit_cube_tetrahedra};
use sonium_fem::physics::electromagnetics::magnetostatics::{
    MU_0, PermeabilityForm, compute_b_field,
};
use sonium_fem::quadrature::QuadratureRule;
use sonium_fem::spaces::{BasisTabulation, FunctionSpace, H1Space};
use sonium_qa::{QaResult, QaStatus};
use sonium_solvers::SparseCholeskyFactorization;

const CASE_ID: &str = "fem.magnetostatics-air-gap.v1";

#[derive(Debug, Clone, Copy)]
struct Slab {
    gap: f64,
    core_mu_r: f64,
    drive: f64,
}

impl Slab {
    fn gap_start(self) -> f64 {
        (1.0 - self.gap) / 2.0
    }
    fn in_gap(self, x: f64) -> bool {
        x > self.gap_start() && x < self.gap_start() + self.gap
    }
    fn mu_r(self, x: f64) -> f64 {
        if self.in_gap(x) { 1.0 } else { self.core_mu_r }
    }
    fn reluctance(self) -> f64 {
        self.gap + (1.0 - self.gap) / self.core_mu_r
    }
    // Ampere's law and normal B continuity: H_gap*g + H_core*(1-g) = drive.
    fn flux(self) -> f64 {
        MU_0 * self.drive / self.reluctance()
    }
    fn potential(self, x: f64) -> f64 {
        let gap_length = (x - self.gap_start()).clamp(0.0, self.gap);
        -self.drive * (gap_length + (x - gap_length) / self.core_mu_r) / self.reluctance()
    }
}

impl BilinearForm<f64> for Slab {
    fn integrate_point(
        &self,
        test: &BasisTabulation,
        trial: &BasisTabulation,
        weight: f64,
        local: &mut LocalMatrix<f64>,
    ) {
        // Normalize the entire equation by mu0. Regional assembly still uses
        // the production permeability integrand on each side of the interface.
        PermeabilityForm::from_mu(self.mu_r(test.physical_coords[0]))
            .integrate_point(test, trial, weight, local);
    }
}

#[derive(Debug, Default)]
struct Errors {
    potential: f64,
    field: f64,
    gap_flux: f64,
    energy: f64,
    gap_energy: f64,
    face_flux: f64,
    interface_flux: f64,
    residual: f64,
    boundary: f64,
    free_dofs: usize,
}

fn solve(n: usize, slab: Slab) -> Result<Errors, String> {
    let mut mesh = unit_cube_tetrahedra(n);
    // Each of the three material layers occupies a third of the logical grid.
    // Warp x only: both physical interfaces remain conforming on every mesh.
    let a = slab.gap_start();
    for node in &mut mesh.nodes {
        let x = node.x;
        node.x = if x <= 1.0 / 3.0 {
            3.0 * a * x
        } else if x <= 2.0 / 3.0 {
            a + 3.0 * slab.gap * (x - 1.0 / 3.0)
        } else {
            a + slab.gap + 3.0 * a * (x - 2.0 / 3.0)
        };
    }
    for face in &mut mesh.boundaries {
        face.marker = if face.nodes.iter().all(|&i| mesh.nodes[i].x.abs() < 1e-12) {
            1
        } else if face
            .nodes
            .iter()
            .all(|&i| (mesh.nodes[i].x - 1.0).abs() < 1e-12)
        {
            2
        } else {
            3
        };
    }
    let space = H1Space::new(PolynomialDegree::P1);
    let matrix = assemble_bilinear(&slab, &space, &space, &mesh, 2);
    let raw = matrix.to_csr();
    let mut system = FemSystem::new(matrix, vec![0.0; mesh.num_nodes()]);
    for (marker, value) in [(1, 0.0), (2, -slab.drive)] {
        try_apply_dirichlet_values(&mut system, &mesh, &space, marker, &|_, _, _| value, None)
            .map_err(|e| e.to_string())?;
    }
    let left = space.boundary_dofs(&mesh, 1);
    let right = space.boundary_dofs(&mesh, 2);
    let mut constrained = vec![false; mesh.num_nodes()];
    for &i in left.iter().chain(&right) {
        constrained[i] = true;
    }
    let free_dofs = constrained.iter().filter(|&&b| !b).count();
    if free_dofs == 0 {
        return Err("air-gap solve has no free unknowns".into());
    }
    // These small SPD systems admit sparse Cholesky. Direct factorization
    // avoids an iterative stopping scale dominated by lifted high-mu BC rows.
    let factor = SparseCholeskyFactorization::factorize(&system.matrix.to_csr())
        .map_err(|e| format!("air-gap Cholesky: {e}"))?;
    let values = factor.solve(&system.rhs).map_err(|e| e.to_string())?;
    if values.iter().any(|v| !v.is_finite()) {
        return Err("air-gap solve returned a nonfinite potential".into());
    }
    let action = raw.matvec(&values);
    let action_norm = action.iter().map(|v| v * v).sum::<f64>().sqrt();
    if action_norm <= 0.0 {
        return Err("air-gap drive produced no boundary flux".into());
    }
    let residual = action
        .iter()
        .enumerate()
        .filter(|(i, _)| !constrained[*i])
        .map(|(_, v)| v * v)
        .sum::<f64>()
        .sqrt()
        / action_norm;
    let boundary = mesh
        .nodes
        .iter()
        .enumerate()
        .filter(|(i, _)| constrained[*i])
        .map(|(i, p)| (values[i] - slab.potential(p.x)).abs() / slab.drive.abs())
        .fold(0.0_f64, f64::max);
    let expected_flux = slab.flux();
    // The raw weak reactions are -B.n; integrate both end faces independently.
    let face_flux = ((left.iter().map(|&i| MU_0 * action[i]).sum::<f64>() - expected_flux)
        .abs()
        .max((right.iter().map(|&i| MU_0 * action[i]).sum::<f64>() + expected_flux).abs()))
        / expected_flux.abs();
    let mut potential_error = 0.0;
    let mut potential_norm = 0.0;
    let mut field_error = 0.0;
    let mut field_norm = 0.0;
    let mut flux_integral = 0.0;
    let mut gap_volume = 0.0;
    let mut energy = 0.0;
    let mut gap_energy = 0.0;
    let mut region_flux = [0.0; 3];
    let mut region_volume = [0.0; 3];
    for element in 0..mesh.num_elements() {
        let dofs = space.local_to_global(&mesh, element);
        let values: Vec<f64> = dofs.iter().map(|&i| values[i]).collect();
        for qp in QuadratureRule::new(mesh.elements[element].element_type, 2).iter() {
            let tab = space.tabulate(&mesh, element, qp.xi(), qp.eta(), qp.zeta());
            let x = tab.physical_coords[0];
            let weight = qp.weight * tab.det_j;
            let mu = MU_0 * slab.mu_r(x);
            let actual = compute_b_field(&values, &tab, mu, &[0.0; 3]);
            let potential = values
                .iter()
                .enumerate()
                .map(|(i, &u)| u * tab.value(i))
                .sum::<f64>();
            let reference = slab.potential(x);
            potential_error += weight * (potential - reference).powi(2);
            potential_norm += weight * reference.powi(2);
            field_error += weight
                * ((actual[0] - expected_flux).powi(2) + actual[1].powi(2) + actual[2].powi(2));
            field_norm += weight * expected_flux.powi(2);
            let density = actual.iter().map(|v| v * v).sum::<f64>() / (2.0 * mu);
            energy += weight * density;
            let region = if x < a {
                0
            } else if slab.in_gap(x) {
                1
            } else {
                2
            };
            region_flux[region] += weight * actual[0];
            region_volume[region] += weight;
            if slab.in_gap(x) {
                flux_integral += weight * actual[0];
                gap_volume += weight;
                gap_energy += weight * density;
            }
        }
    }
    if region_volume.iter().any(|v| *v <= 0.0) {
        return Err("air-gap quadrature omitted a material layer".into());
    }
    let interface_flux = (0..2)
        .map(|i| {
            (region_flux[i] / region_volume[i] - region_flux[i + 1] / region_volume[i + 1]).abs()
                / expected_flux.abs()
        })
        .fold(0.0_f64, f64::max);
    let expected_energy = 0.5 * MU_0 * slab.drive.powi(2) / slab.reluctance();
    let expected_gap_energy = expected_flux.powi(2) * slab.gap / (2.0 * MU_0);
    Ok(Errors {
        potential: (potential_error / potential_norm).sqrt(),
        field: (field_error / field_norm).sqrt(),
        gap_flux: (flux_integral / gap_volume - expected_flux).abs() / expected_flux.abs(),
        energy: (energy - expected_energy).abs() / expected_energy,
        gap_energy: (gap_energy - expected_gap_energy).abs() / expected_gap_energy,
        face_flux,
        interface_flux,
        residual,
        boundary,
        free_dofs,
    })
}

fn run() -> Result<QaResult, String> {
    // Retain the original C-core tag fixture separately from the solved slab.
    // The exact-reference slab tests material interfaces; it does not claim
    // a fringing-field solution for the C-core geometry.
    let core = c_core_mesh_triangles(CCoreMeshConfig {
        width: 1.0,
        height: 0.8,
        core_thickness: 0.18,
        gap_height: 0.12,
        nx: 12,
        ny: 10,
    });
    if !core.element_regions.contains(&CCoreRegion::Coil)
        || !core.element_regions.contains(&CCoreRegion::Gap)
    {
        return Err("C-core region fixture lost coil/gap tags".into());
    }
    let mut worst = Errors::default();
    let mut minimum_free_dofs = usize::MAX;
    let mut cases = 0;
    for gap in [0.05, 0.2] {
        for core_mu_r in [1.0, 100.0, 1000.0] {
            for drive in [-1000.0, 1000.0] {
                for n in [3, 6, 12] {
                    let e = solve(
                        n,
                        Slab {
                            gap,
                            core_mu_r,
                            drive,
                        },
                    )?;
                    println!("air gap={gap} mu_r={core_mu_r} drive={drive} n={n}: {e:?}");
                    worst.potential = worst.potential.max(e.potential);
                    worst.field = worst.field.max(e.field);
                    worst.gap_flux = worst.gap_flux.max(e.gap_flux);
                    worst.energy = worst.energy.max(e.energy);
                    worst.gap_energy = worst.gap_energy.max(e.gap_energy);
                    worst.face_flux = worst.face_flux.max(e.face_flux);
                    worst.interface_flux = worst.interface_flux.max(e.interface_flux);
                    worst.residual = worst.residual.max(e.residual);
                    worst.boundary = worst.boundary.max(e.boundary);
                    minimum_free_dofs = minimum_free_dofs.min(e.free_dofs);
                    cases += 1;
                }
            }
        }
    }
    let mut result = QaResult::passed(CASE_ID)
        .requested_backend("cpu")
        .actual_backend("cpu");
    for (name, error, bound) in [
        ("potential_error", worst.potential, 1e-7),
        ("magnetic_field_error", worst.field, 1e-7),
        ("gap_flux_error", worst.gap_flux, 1e-7),
        ("total_energy_error", worst.energy, 1e-7),
        ("gap_energy_error", worst.gap_energy, 1e-7),
        ("end_face_flux_error", worst.face_flux, 1e-7),
        ("interface_flux_jump", worst.interface_flux, 1e-7),
        ("physical_residual", worst.residual, 1e-8),
        ("boundary_error", worst.boundary, 1e-9),
    ] {
        if !error.is_finite() || error > bound {
            return Err(format!("{name}: {error:.3e} > {bound:.3e}"));
        }
        result = result.metric(name, error).maximum(name, bound);
    }
    Ok(result.metric("solved_cases",cases as f64).minimum("solved_cases",36.0)
        .metric("minimum_free_dofs",minimum_free_dofs as f64).minimum("minimum_free_dofs",1.0)
        .seed("air-gap:g0.05-0.2:mur1-100-1000:drive+-1000:n3-6-12")
        .input_checksum("conforming-tetra-core-air-core:36-solves:v1")
        .reference("series reluctance R=g/mu0+(1-g)/(mu0*mur), flux=drive/R, W=drive^2/(2R), W_gap=B^2*g/(2mu0); exact piecewise potential")
        .reference_checksum("analytic:core-air-core-series-reluctance:v1")
        .dimension("3D unit slab, core-air-core interfaces, essential end potentials and natural zero side flux")
        .formulation("production regional permeability form, full P1 tetrahedral sparse Cholesky solve, B recovery, quadrature energy and raw boundary reactions")
        .tolerance("all 36 solved potential/field/flux/energy cases <=1e-7; true residual<=1e-8"))
}

fn main() {
    let result = match run() {
        Ok(r) => r,
        Err(e) => QaResult::failed(CASE_ID, e),
    };
    match result.to_json_line() {
        Ok(json) => println!("QA_RESULT: {json}"),
        Err(e) => {
            eprintln!("FAILED: {e}");
            std::process::exit(1);
        }
    }
    if result.status == QaStatus::Passed {
        println!("SUCCESS");
    } else {
        println!("FAILED");
        std::process::exit(1);
    }
}
