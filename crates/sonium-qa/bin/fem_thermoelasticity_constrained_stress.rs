//! Solved thermal stress, transverse expansion and support-reaction patches.

use sonium_fem::assembly::TripletMatrix;
use sonium_fem::basis::PolynomialDegree;
use sonium_fem::mesh::unit_cube_tetrahedra;
use sonium_fem::physics::coupled::thermoelasticity::{
    ThermoelasticMaterial, assemble_thermal_strain_load, assemble_thermoelastic_stiffness,
};
use sonium_fem::physics::solid::elasticity::IsotropicMaterial;
use sonium_fem::quadrature::QuadratureRule;
use sonium_fem::spaces::{FunctionSpace, H1Space};
use sonium_qa::{QaResult, QaStatus};
use sonium_solvers::direct::SparseLuFactorization;

const CASE_ID: &str = "fem.thermoelasticity-constrained-stress.v1";

#[derive(Debug, Default)]
struct Metrics {
    displacement_error: f64,
    stress_error: f64,
    reaction_error: f64,
    residual: f64,
    boundary_error: f64,
    maximum_transverse_displacement: f64,
    minimum_free_dofs: usize,
    solved_cases: usize,
}

fn solve(
    n: usize,
    material: ThermoelasticMaterial,
    delta_temperature: f64,
    fully_blocked: bool,
) -> Result<Metrics, String> {
    let mesh = unit_cube_tetrahedra(n);
    let degree = PolynomialDegree::P1;
    let elastic = IsotropicMaterial::new(material.young_modulus, material.poisson_ratio);
    let temperature = material.reference_temperature + delta_temperature;
    let physical_matrix =
        assemble_thermoelastic_stiffness(&mesh, degree, 3, elastic, None).to_csr();
    let physical_rhs = assemble_thermal_strain_load(&mesh, degree, 3, material, temperature);
    let count = physical_rhs.len();
    if physical_rhs.iter().all(|x| x.abs() == 0.0) {
        return Err("thermal strain load is empty".into());
    }
    let mut constrained = vec![false; count];
    for (node, p) in mesh.nodes.iter().enumerate() {
        let boundary = [p.x, p.y, p.z]
            .iter()
            .any(|v| v.abs() < 1e-12 || (v - 1.0).abs() < 1e-12);
        if fully_blocked {
            if boundary {
                constrained[3 * node..3 * node + 3].fill(true);
            }
        } else {
            // End faces constrain axial expansion only. Transverse symmetry
            // planes remove rigid motion while allowing real Poisson/thermal
            // expansion on all other faces.
            constrained[3 * node] = p.x.abs() < 1e-12 || (p.x - 1.0).abs() < 1e-12;
            constrained[3 * node + 1] = p.y.abs() < 1e-12;
            constrained[3 * node + 2] = p.z.abs() < 1e-12;
        }
    }
    let free_dofs = constrained.iter().filter(|v| !**v).count();
    if free_dofs == 0 {
        return Err("thermal patch has no free mechanical unknowns".into());
    }
    // Scale the entire physical equation by E, then enforce homogeneous BCs.
    let mut matrix = TripletMatrix::new(count, count);
    let mut rhs = vec![0.0; count];
    for row in 0..count {
        if constrained[row] {
            matrix.push(row, row, 1.0);
        } else {
            rhs[row] = physical_rhs[row] / material.young_modulus;
            for (column, value) in physical_matrix.row_entries(row) {
                if !constrained[column] {
                    matrix.push(row, column, value / material.young_modulus);
                }
            }
        }
    }
    let factor = SparseLuFactorization::factorize(&matrix.to_csr()).map_err(|e| e.to_string())?;
    let values = factor.solve(&rhs).map_err(|e| e.to_string())?;
    if values.iter().any(|v| !v.is_finite()) {
        return Err("nonfinite thermal displacement".into());
    }
    let action = physical_matrix.matvec(&values);
    let residual = action
        .iter()
        .zip(&physical_rhs)
        .enumerate()
        .filter(|(i, _)| !constrained[*i])
        .map(|(_, (&a, &b))| (a - b).powi(2))
        .sum::<f64>()
        .sqrt()
        / physical_rhs.iter().map(|v| v * v).sum::<f64>().sqrt();
    let boundary_error = values
        .iter()
        .enumerate()
        .filter(|(i, _)| constrained[*i])
        .map(|(_, v)| v.abs())
        .fold(0.0_f64, f64::max);
    let thermal_strain = material.thermal_expansion * delta_temperature;
    let expected_stress = if fully_blocked {
        -material.young_modulus * thermal_strain / (1.0 - 2.0 * material.poisson_ratio)
    } else {
        -material.young_modulus * thermal_strain
    };
    let expansion = if fully_blocked {
        0.0
    } else {
        (1.0 + material.poisson_ratio) * thermal_strain
    };
    // Reaction is recovered from the unmodified thermal equation, not the
    // unit Dirichlet rows. Opposing unit-area faces carry opposite forces.
    let mut reaction_error = 0.0_f64;
    let axes = if fully_blocked { 3 } else { 1 };
    for axis in 0..axes {
        for (coordinate, sign) in [(0.0, -1.0), (1.0, 1.0)] {
            let reaction = mesh
                .nodes
                .iter()
                .enumerate()
                .filter(|(_, p)| ([p.x, p.y, p.z][axis] - coordinate).abs() < 1e-12)
                .map(|(node, _)| action[3 * node + axis] - physical_rhs[3 * node + axis])
                .sum::<f64>();
            reaction_error = reaction_error
                .max((reaction - sign * expected_stress).abs() / expected_stress.abs());
        }
    }
    let scalar_space = H1Space::new(degree);
    let constitutive = elastic.voigt_3d();
    let (mut displacement_error, mut stress_error, mut maximum_transverse_displacement) =
        (0.0, 0.0, 0.0_f64);
    for element in 0..mesh.num_elements() {
        let dofs = scalar_space.local_to_global(&mesh, element);
        for qp in QuadratureRule::new(mesh.elements[element].element_type, 2).iter() {
            let tab = scalar_space.tabulate(&mesh, element, qp.xi(), qp.eta(), qp.zeta());
            let mut displacement = [0.0; 3];
            let mut gradient = [[0.0; 3]; 3];
            for (local, &node) in dofs.iter().enumerate() {
                for component in 0..3 {
                    displacement[component] += values[3 * node + component] * tab.value(local);
                    for dimension in 0..3 {
                        gradient[component][dimension] +=
                            values[3 * node + component] * tab.grad(local, dimension);
                    }
                }
            }
            maximum_transverse_displacement = maximum_transverse_displacement
                .max(displacement[1].abs())
                .max(displacement[2].abs());
            let expected = [
                0.0,
                expansion * tab.physical_coords[1],
                expansion * tab.physical_coords[2],
            ];
            let strain = [
                gradient[0][0] - thermal_strain,
                gradient[1][1] - thermal_strain,
                gradient[2][2] - thermal_strain,
                gradient[1][2] + gradient[2][1],
                gradient[0][2] + gradient[2][0],
                gradient[0][1] + gradient[1][0],
            ];
            let stress: [f64; 6] = std::array::from_fn(|row| {
                (0..6)
                    .map(|column| constitutive[6 * row + column] * strain[column])
                    .sum()
            });
            let stress_reference = if fully_blocked {
                [
                    expected_stress,
                    expected_stress,
                    expected_stress,
                    0.0,
                    0.0,
                    0.0,
                ]
            } else {
                [expected_stress, 0.0, 0.0, 0.0, 0.0, 0.0]
            };
            let weight = qp.weight * tab.det_j;
            displacement_error += weight
                * (0..3)
                    .map(|d| (displacement[d] - expected[d]).powi(2))
                    .sum::<f64>()
                / thermal_strain.powi(2);
            stress_error += weight
                * (0..6)
                    .map(|d| (stress[d] - stress_reference[d]).powi(2))
                    .sum::<f64>()
                / expected_stress.powi(2);
        }
    }
    Ok(Metrics {
        displacement_error: displacement_error.sqrt(),
        stress_error: stress_error.sqrt(),
        reaction_error,
        residual,
        boundary_error,
        maximum_transverse_displacement,
        minimum_free_dofs: free_dofs,
        solved_cases: 1,
    })
}

fn run() -> Result<QaResult, String> {
    let mut total = Metrics {
        minimum_free_dofs: usize::MAX,
        ..Default::default()
    };
    for (e, nu, alpha) in [(210e9, 0.3, 12e-6), (70e9, 0.33, 23e-6), (3e9, 0.35, 50e-6)] {
        let material = ThermoelasticMaterial {
            young_modulus: e,
            poisson_ratio: nu,
            thermal_expansion: alpha,
            reference_temperature: 293.0,
        };
        for delta in [-50.0, 100.0] {
            for n in [2, 4] {
                for blocked in [false, true] {
                    let m = solve(n, material, delta, blocked)?;
                    println!("thermal E={e:.2e} dT={delta} n={n} full={blocked}: {m:?}");
                    total.displacement_error = total.displacement_error.max(m.displacement_error);
                    total.stress_error = total.stress_error.max(m.stress_error);
                    total.reaction_error = total.reaction_error.max(m.reaction_error);
                    total.residual = total.residual.max(m.residual);
                    total.boundary_error = total.boundary_error.max(m.boundary_error);
                    total.maximum_transverse_displacement = total
                        .maximum_transverse_displacement
                        .max(m.maximum_transverse_displacement);
                    total.minimum_free_dofs = total.minimum_free_dofs.min(m.minimum_free_dofs);
                    total.solved_cases += m.solved_cases;
                }
            }
        }
    }
    let mut record = QaResult::passed(CASE_ID)
        .requested_backend("cpu")
        .actual_backend("cpu");
    for (key, value, bound) in [
        ("displacement_error", total.displacement_error, 1e-9),
        ("stress_error", total.stress_error, 1e-9),
        ("reaction_error", total.reaction_error, 1e-9),
        ("physical_residual", total.residual, 1e-10),
        ("boundary_error", total.boundary_error, 1e-12),
    ] {
        if !value.is_finite() || value > bound {
            return Err(format!("{key} {value:.3e} exceeds {bound:.3e}"));
        }
        record = record.metric(key, value).maximum(key, bound);
    }
    if total.maximum_transverse_displacement < 1e-4 {
        return Err("axially constrained thermal solve produced no transverse expansion".into());
    }
    Ok(record.metric("maximum_transverse_displacement",total.maximum_transverse_displacement).minimum("maximum_transverse_displacement",1e-4)
        .metric("minimum_free_dofs",total.minimum_free_dofs as f64).minimum("minimum_free_dofs",1.0)
        .metric("solved_cases",total.solved_cases as f64).minimum("solved_cases",24.0)
        .reference("uniaxially constrained sigma_x=-E alpha dT and transverse strain=(1+nu) alpha dT; fully blocked hydrostatic sigma=-E alpha dT/(1-2nu)")
        .reference_checksum("analytic:isotropic-thermal-constraint-patches:v1")
        .input_checksum("unit-cube:n2-n4:steel-aluminum-polymer:dT-minus50-plus100:uniaxial-hydrostatic:v1")
        .seed("thermal-constraint:3-materials:2-temperatures:2-meshes:2-boundary-cases")
        .dimension("3D tetrahedral unit cube with nonzero thermal strain and free mechanical DOFs")
        .formulation("production thermoelastic stiffness and thermal load; solved displacement, constitutive stress and physical support reactions")
        .tolerance("exact affine P1 patch solutions; all displacement/stress/reaction errors <= 1e-9 and true residual <= 1e-10"))
}

fn main() {
    let record = match run() {
        Ok(record) => record,
        Err(error) => QaResult::failed(CASE_ID, error),
    };
    match record.to_json_line() {
        Ok(json) => println!("QA_RESULT: {json}"),
        Err(error) => {
            eprintln!("FAILED: {error}");
            std::process::exit(1);
        }
    }
    if record.status != QaStatus::Passed {
        println!("FAILED");
        std::process::exit(1);
    }
    println!("SUCCESS");
}
