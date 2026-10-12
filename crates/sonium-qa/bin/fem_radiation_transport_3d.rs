//! Radiation streaming contract and solved 3D grey S2 transport reference.
//!
//! Native angular scattering and spatially varying prescribed-temperature
//! emission are exercised. This is steady isotropic-scattering S2 transport;
//! it does not claim angular refinement or coupled thermal time integration.

use sonium_fem::assembly::{TripletMatrix, assemble_bilinear, assemble_linear};
use sonium_fem::basis::PolynomialDegree;
use sonium_fem::boundary::generic::{FemSystem, try_apply_dirichlet_values};
use sonium_fem::mesh::{Mesh, unit_cube_tetrahedra};
use sonium_fem::physics::thermal::heat_conduction::HeatSourceForm;
use sonium_fem::physics::thermal::radiation_transport::{
    AngularQuadrature, RadiativeMaterial, StreamingExtinctionForm, ThermalEmissionForm,
    assemble_discrete_ordinates, radiative_material_heating,
};
use sonium_fem::quadrature::QuadratureRule;
use sonium_fem::spaces::{FunctionSpace, H1Space};
use sonium_qa::{QaResult, QaStatus};
use sonium_solvers::{CsrMatrix, SparseLuFactorization};
use std::f64::consts::PI;

const CASE: &str = "fem.radiation-transport-3d-s2-mms.v1";
const MATERIAL: RadiativeMaterial = RadiativeMaterial {
    absorption_coefficient: 2.0,
    scattering_coefficient: 1.5,
    asymmetry_parameter: 0.0,
    refractive_index: 1.0,
    density: 1.0,
    specific_heat: 1.0,
};
fn base(x: f64, y: f64, z: f64) -> f64 {
    2.0 + x * x + 2.0 * y * y + 3.0 * z * z + x * y + y * z + 2.0 * x * z
}
fn radiance(x: f64, y: f64, z: f64) -> f64 {
    1.0 + 0.2 * (x * x + y * y + z * z)
}
fn temperature(x: f64, y: f64, z: f64) -> f64 {
    (PI * radiance(x, y, z) / 5.670_374_419e-8).powf(0.25)
}
fn reference(s: [f64; 3], x: f64, y: f64, z: f64) -> (f64, [f64; 3]) {
    (
        base(x, y, z)
            + 0.2
                * (s[0] * (1.0 + x + y * z) + s[1] * (2.0 + y + x * z) + s[2] * (3.0 + z + x * y)),
        [
            2.0 * x + y + 2.0 * z + 0.2 * (s[0] + s[1] * z + s[2] * y),
            4.0 * y + x + z + 0.2 * (s[0] * z + s[1] + s[2] * x),
            6.0 * z + y + 2.0 * x + 0.2 * (s[0] * y + s[1] * x + s[2]),
        ],
    )
}
fn forcing(s: [f64; 3], x: f64, y: f64, z: f64) -> f64 {
    let (intensity, grad) = reference(s, x, y, z);
    s.iter().zip(grad).map(|(a, b)| a * b).sum::<f64>() + 3.5 * intensity
        - 1.5 * base(x, y, z)
        - 2.0 * radiance(x, y, z)
}
#[derive(Debug, Clone, Copy)]
enum Control {
    Full,
    NoScatteringGain,
    NoEmission,
    NoAbsorption,
    NoZStreaming,
}
fn boundary_system(
    matrix: TripletMatrix<f64>,
    rhs: Vec<f64>,
    mesh: &Mesh,
    space: &H1Space,
    direction: [f64; 3],
) -> Result<FemSystem<f64>, String> {
    let mut system = FemSystem::new(matrix, rhs);
    for (axis, &component) in direction.iter().enumerate() {
        let tag = 2 * axis + if component > 0.0 { 1 } else { 2 };
        try_apply_dirichlet_values(
            &mut system,
            mesh,
            space,
            tag as i32,
            &|x, y, z| reference(direction, x, y, z).0,
            None,
        )
        .map_err(|e| e.to_string())?;
    }
    Ok(system)
}
#[derive(Debug)]
struct Metrics {
    intensity: f64,
    gradient: f64,
    incident: f64,
    heating: f64,
    residual: f64,
    boundary: f64,
    energy_balance: f64,
    flux_error: f64,
    iterations: usize,
    free: usize,
}
fn solve(n: usize, degree: PolynomialDegree, control: Control) -> Result<Metrics, String> {
    let mut mesh = unit_cube_tetrahedra(n);
    for face in &mut mesh.boundaries {
        let mut found = false;
        for axis in 0..3 {
            for side in 0..2 {
                if face.nodes.iter().all(|&i| {
                    let p = &mesh.nodes[i];
                    let coordinate = [p.x, p.y, p.z][axis];
                    (coordinate - side as f64).abs() < 1e-12
                }) {
                    face.marker = (2 * axis + side + 1) as i32;
                    found = true;
                }
            }
        }
        if !found {
            return Err("cube boundary is not a coordinate plane".into());
        }
    }
    let space = H1Space::new(degree);
    let count = space.num_global_dofs(&mesh);
    let angular = AngularQuadrature::s2_3d();
    angular.validate().map_err(|e| format!("{e:?}"))?;
    let material = if matches!(control, Control::NoAbsorption) {
        RadiativeMaterial {
            absorption_coefficient: 0.0,
            ..MATERIAL
        }
    } else {
        MATERIAL
    };
    material.validate().map_err(|e| format!("{e:?}"))?;
    let mut native = assemble_discrete_ordinates(&mesh, degree, material, &angular);
    if matches!(control, Control::NoScatteringGain) {
        // Preserve full extinction while removing all scattering gain blocks.
        for outgoing in 0..8 {
            for incoming in 0..8 {
                if outgoing == incoming {
                    native.blocks[outgoing][incoming] = assemble_bilinear(
                        &StreamingExtinctionForm {
                            direction: angular.directions[outgoing],
                            extinction_coefficient: 3.5,
                        },
                        &space,
                        &space,
                        &mesh,
                        4,
                    );
                } else {
                    native.blocks[outgoing][incoming] = TripletMatrix::new(count, count);
                }
            }
        }
    }
    if matches!(control, Control::NoZStreaming) {
        for outgoing in 0..8 {
            let correction = assemble_bilinear(
                &StreamingExtinctionForm {
                    direction: [0.0, 0.0, -angular.directions[outgoing][2]],
                    extinction_coefficient: 0.0,
                },
                &space,
                &space,
                &mesh,
                4,
            );
            for k in 0..correction.nnz() {
                native.blocks[outgoing][outgoing].push(
                    correction.rows[k],
                    correction.cols[k],
                    correction.values[k],
                );
            }
        }
    }
    let emission = if matches!(control, Control::NoEmission) {
        vec![0.0; count]
    } else {
        assemble_linear(
            &ThermalEmissionForm {
                material,
                temperature,
            },
            &space,
            &mesh,
            4,
        )
    };
    let rhs: Vec<Vec<f64>> = angular
        .directions
        .iter()
        .map(|&s| {
            let source = assemble_linear(
                &HeatSourceForm {
                    source_fn: move |x, y, z| forcing(s, x, y, z),
                },
                &space,
                &mesh,
                4,
            );
            source.iter().zip(&emission).map(|(a, b)| a + b).collect()
        })
        .collect();
    let mut fixed = vec![vec![false; count]; 8];
    for (outgoing, s) in angular.directions.iter().enumerate() {
        for (axis, &component) in s.iter().enumerate() {
            let tag = (2 * axis + if component > 0.0 { 1 } else { 2 }) as i32;
            for d in space.boundary_dofs(&mesh, tag) {
                fixed[outgoing][d] = true;
            }
        }
    }
    let free = fixed
        .iter()
        .map(|f| f.iter().filter(|v| !**v).count())
        .min()
        .unwrap_or(0);
    if free == 0 {
        return Err("angular transport has no free unknowns".into());
    }
    let factors: Vec<_> = (0..8)
        .map(|a| {
            SparseLuFactorization::factorize(
                &boundary_system(
                    native.blocks[a][a].clone(),
                    vec![0.0; count],
                    &mesh,
                    &space,
                    angular.directions[a],
                )?
                .matrix
                .to_csr(),
            )
            .map_err(|e| e.to_string())
        })
        .collect::<Result<_, String>>()?;
    let blocks: Vec<Vec<CsrMatrix<f64>>> = native
        .blocks
        .iter()
        .map(|r| r.iter().map(TripletMatrix::to_csr).collect())
        .collect();
    let mut values = vec![vec![1.0; count]; 8];
    let mut iterations = 0;
    let mut converged = false;
    for iteration in 1..=150 {
        let mut next = Vec::new();
        let mut change = 0.0_f64;
        for outgoing in 0..8 {
            let mut load = rhs[outgoing].clone();
            for incoming in 0..8 {
                if incoming != outgoing {
                    let action = blocks[outgoing][incoming].matvec(&values[incoming]);
                    for (r, a) in load.iter_mut().zip(action) {
                        *r -= a;
                    }
                }
            }
            let system = boundary_system(
                native.blocks[outgoing][outgoing].clone(),
                load,
                &mesh,
                &space,
                angular.directions[outgoing],
            )?;
            let state = factors[outgoing]
                .solve(&system.rhs)
                .map_err(|e| e.to_string())?;
            if state.iter().any(|v| !v.is_finite()) {
                return Err("nonfinite solved intensity".into());
            }
            let error: f64 = state
                .iter()
                .zip(&values[outgoing])
                .map(|(a, b)| (a - b).powi(2))
                .sum();
            let norm: f64 = state.iter().map(|v| v * v).sum();
            change = change.max((error / norm).sqrt());
            next.push(state);
        }
        values = next;
        iterations = iteration;
        if change < 1e-12 {
            converged = true;
            break;
        }
    }
    if !converged {
        return Err("angular source iteration did not converge".into());
    }
    // Positivity is required for the physical reference problem. Controls
    // hold its signed manufactured forcing fixed while removing terms, so
    // their deliberately incorrect operators may return negative intensities.
    if matches!(control, Control::Full) && values.iter().flatten().any(|v| *v <= 0.0) {
        return Err("nonpositive converged reference intensity".into());
    }
    let mut residual = 0.0_f64;
    for outgoing in 0..8 {
        let mut action = vec![0.0; count];
        for incoming in 0..8 {
            for (a, b) in action
                .iter_mut()
                .zip(blocks[outgoing][incoming].matvec(&values[incoming]))
            {
                *a += b;
            }
        }
        let error: f64 = (0..count)
            .filter(|&d| !fixed[outgoing][d])
            .map(|d| (action[d] - rhs[outgoing][d]).powi(2))
            .sum();
        let norm: f64 = (0..count)
            .filter(|&d| !fixed[outgoing][d])
            .map(|d| rhs[outgoing][d].powi(2))
            .sum();
        residual = residual.max((error / norm).sqrt());
    }
    let coords = space
        .dof_coordinates(&mesh)
        .ok_or("missing H1 coordinates")?;
    let mut boundary = 0.0_f64;
    for outgoing in 0..8 {
        for (d, p) in coords.iter().enumerate() {
            if fixed[outgoing][d] {
                boundary = boundary.max(
                    (values[outgoing][d]
                        - reference(angular.directions[outgoing], p[0], p[1], p[2]).0)
                        .abs(),
                );
            }
        }
    }
    let mut error = 0.0;
    let mut norm = 0.0;
    let mut grad_error = 0.0;
    let mut grad_norm = 0.0;
    let mut incident_error = 0.0;
    let mut incident_norm = 0.0;
    let mut heat_error = 0.0;
    let mut heat_norm = 0.0;
    let mut heating = 0.0;
    let mut outward_flux = 0.0;
    let mut external = 0.0;
    for element in 0..mesh.num_elements() {
        let dofs = space.local_to_global(&mesh, element);
        for qp in QuadratureRule::new(mesh.elements[element].element_type, 5).iter() {
            let tab = space.tabulate(&mesh, element, qp.xi(), qp.eta(), qp.zeta());
            let [x, y, z] = tab.physical_coords;
            let w = qp.weight * tab.det_j;
            let mut incident = 0.0;
            for outgoing in 0..8 {
                let s = angular.directions[outgoing];
                let wa = angular.weights[outgoing];
                let (exact, grad) = reference(s, x, y, z);
                let value: f64 = dofs
                    .iter()
                    .enumerate()
                    .map(|(i, &d)| values[outgoing][d] * tab.value(i))
                    .sum();
                error += w * wa * (value - exact).powi(2);
                norm += w * wa * exact.powi(2);
                incident += wa * value;
                for axis in 0..3 {
                    let derivative: f64 = dofs
                        .iter()
                        .enumerate()
                        .map(|(i, &d)| values[outgoing][d] * tab.grad(i, axis))
                        .sum();
                    grad_error += w * wa * (derivative - grad[axis]).powi(2);
                    grad_norm += w * wa * grad[axis].powi(2);
                    outward_flux += w * wa * s[axis] * derivative;
                }
                external += w * wa * forcing(s, x, y, z);
            }
            let expected_g = 4.0 * PI * base(x, y, z);
            let expected_heat = 2.0 * 4.0 * PI * (base(x, y, z) - radiance(x, y, z));
            let heat = radiative_material_heating(MATERIAL, incident, temperature(x, y, z));
            incident_error += w * (incident - expected_g).powi(2);
            incident_norm += w * expected_g.powi(2);
            heat_error += w * (heat - expected_heat).powi(2);
            heat_norm += w * expected_heat.powi(2);
            heating += w * heat;
        }
    }
    Ok(Metrics {
        intensity: (error / norm).sqrt(),
        gradient: (grad_error / grad_norm).sqrt(),
        incident: (incident_error / incident_norm).sqrt(),
        heating: (heat_error / heat_norm).sqrt(),
        residual,
        boundary,
        energy_balance: (outward_flux + heating - external).abs() / external.abs(),
        flux_error: (outward_flux - 0.8 * PI).abs() / (0.8 * PI),
        iterations,
        free,
    })
}
fn run() -> Result<QaResult, String> {
    let mut levels = Vec::new();
    for n in [3, 6, 12] {
        let m = solve(n, PolynomialDegree::P1, Control::Full)?;
        println!("radiation S2 P1 n={n}: {m:?}");
        levels.push(m);
    }
    let patch = solve(2, PolynomialDegree::P2, Control::Full)?;
    println!("radiation S2 P2 exact: {patch:?}");
    let mut negative = f64::INFINITY;
    for control in [
        Control::NoScatteringGain,
        Control::NoEmission,
        Control::NoAbsorption,
        Control::NoZStreaming,
    ] {
        let m = solve(2, PolynomialDegree::P2, control)?;
        println!("radiation S2 {control:?}: {m:?}");
        negative = negative.min(m.intensity);
    }
    let value_rate = levels
        .windows(2)
        .map(|m| (m[0].intensity / m[1].intensity).log2())
        .fold(f64::INFINITY, f64::min);
    let grad_rate = levels
        .windows(2)
        .map(|m| (m[0].gradient / m[1].gradient).log2())
        .fold(f64::INFINITY, f64::min);
    let mut result = QaResult::passed(CASE)
        .requested_backend("cpu")
        .actual_backend("cpu");
    for (name, value, bound) in [
        ("fine_intensity_error", levels[2].intensity, 0.01),
        ("fine_intensity_gradient_error", levels[2].gradient, 0.1),
        ("fine_incident_radiation_error", levels[2].incident, 0.01),
        ("fine_material_heating_error", levels[2].heating, 0.02),
        ("fine_energy_balance", levels[2].energy_balance, 0.01),
        ("fine_boundary_flux_error", levels[2].flux_error, 0.1),
        (
            "p2_exact_field_error",
            patch
                .intensity
                .max(patch.gradient)
                .max(patch.incident)
                .max(patch.heating),
            1e-9,
        ),
        (
            "p2_exact_energy_balance",
            patch.energy_balance.max(patch.flux_error),
            1e-9,
        ),
        (
            "fresh_angular_residual",
            levels
                .iter()
                .chain([&patch])
                .map(|m| m.residual)
                .fold(0.0_f64, f64::max),
            1e-9,
        ),
        (
            "inflow_boundary_error",
            levels
                .iter()
                .chain([&patch])
                .map(|m| m.boundary)
                .fold(0.0_f64, f64::max),
            1e-10,
        ),
    ] {
        if !value.is_finite() || value > bound {
            return Err(format!("{name}={value:.6e} exceeds {bound}"));
        }
        result = result.metric(name, value).maximum(name, bound);
    }
    for (name, value, bound) in [
        ("intensity_spatial_rate", value_rate, 1.7),
        ("gradient_spatial_rate", grad_rate, 0.9),
        ("minimum_omitted_term_error", negative, 1e-3),
    ] {
        if !value.is_finite() || value < bound {
            return Err(format!("{name}={value:.6e} below {bound}"));
        }
        result = result.metric(name, value).minimum(name, bound);
    }
    Ok(result.metric("minimum_free_dofs_per_angle",levels.iter().chain([&patch]).map(|m|m.free).min().unwrap_or(0) as f64).minimum("minimum_free_dofs_per_angle",1.0)
        .metric("executed_angular_solves",8.0*levels.iter().chain([&patch]).map(|m|m.iterations).sum::<usize>() as f64).minimum("executed_angular_solves",32.0)
        .metric("negative_controls",4.0).minimum("negative_controls",4.0)
        .reference("independent quadratic intensity F+0.2*s.(1+x+yz,2+y+xz,3+z+xy), G=4piF, H divergence=0.8pi, prescribed blackbody radiance; inflow-only data")
        .reference_checksum("analytic:3d-grey-isotropic-S2-steady-transport:v1")
        .input_checksum("P1n3-6-12:P2n2:eight-octants:ka2:ks1.5:blackbody-spatial:four-controls")
        .dimension("3D tetrahedral cube, eight coupled S2 angles, all xyz streaming and mixed polynomial fields, three inflow faces per angle")
        .formulation("native discrete-ordinates angular blocks and thermal emission; cached sparse-LU source iteration; actual FE incident radiation/material heating and flux divergence")
        .tolerance("value/gradient rates>=1.7/0.9; P2 fields/energy<=1e-9; residual<=1e-9; fine energy<=0.01; missing terms>=1e-3"))
}
fn main() {
    sonium_qa::fem_physics_3d::run_bilinear_case(
        "fem.radiation-transport-3d.v1",
        &StreamingExtinctionForm {
            direction: [1.0, 0.0, 0.0],
            extinction_coefficient: 1.0,
        },
        false,
    );
    let result = match run() {
        Ok(r) => r,
        Err(e) => QaResult::failed(CASE, e),
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
        std::process::exit(1);
    }
}
