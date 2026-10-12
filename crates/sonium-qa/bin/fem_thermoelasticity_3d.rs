//! Thermoelastic form contract and transient thermal/quasi-static 3D MMS.
//!
//! The caller composes C*theta_t-k*lap(theta)+gamma*theta=Q and elastic
//! equilibrium with thermal strain. The native scalar mass-like coupling is
//! used as the declared thermal relaxation term gamma*theta. Mechanical
//! inertia and reciprocal strain-rate heating are outside this staggered model.

use sonium_fem::assembly::{TripletMatrix, assemble_linear};
use sonium_fem::basis::PolynomialDegree;
use sonium_fem::boundary::generic::{FemSystem, try_apply_dirichlet_values};
use sonium_fem::forms::{LinearForm, LocalVector};
use sonium_fem::mesh::{Mesh, unit_cube_tetrahedra};
use sonium_fem::physics::coupled::thermoelasticity::{
    ThermalStrainLoadForm, ThermoelasticDampingForm, ThermoelasticMaterial,
    build_thermoelastic_coupled_blocks,
};
use sonium_fem::physics::solid::elasticity::{BodyForceForm, IsotropicMaterial};
use sonium_fem::physics::thermal::heat_conduction::{HeatSourceForm, ThermalMaterial};
use sonium_fem::quadrature::QuadratureRule;
use sonium_fem::spaces::{BasisTabulation, FunctionSpace, H1Space, VectorH1Space};
use sonium_qa::{QaResult, QaStatus};
use sonium_solvers::{CsrMatrix, SparseCholeskyFactorization};
use std::collections::HashMap;

const CASE: &str = "fem.thermoelasticity-3d-history-mms.v1";
const MATERIAL: ThermoelasticMaterial = ThermoelasticMaterial {
    young_modulus: 100.0,
    poisson_ratio: 0.25,
    thermal_expansion: 0.001,
    reference_temperature: 293.15,
};
#[derive(Debug, Clone, Copy)]
enum TimeLaw {
    Linear,
    Exponential,
}
impl TimeLaw {
    fn value(self, t: f64) -> f64 {
        match self {
            Self::Linear => 1.0 + t,
            Self::Exponential => (-t).exp(),
        }
    }
    fn derivative(self, t: f64) -> f64 {
        match self {
            Self::Linear => 1.0,
            Self::Exponential => -(-t).exp(),
        }
    }
}
#[derive(Debug, Clone, Copy)]
enum Control {
    Full,
    NoCapacity,
    NoRelaxation,
    NoThermalLoad,
    NoHeatSource,
}
fn reference(x: f64, y: f64, z: f64) -> (f64, [f64; 3], [f64; 3], [[f64; 3]; 3]) {
    (
        0.2 * (1.0 + x * x + 2.0 * y * y + 3.0 * z * z + x * y + y * z + 2.0 * x * z),
        [
            0.2 * (2.0 * x + y + 2.0 * z),
            0.2 * (4.0 * y + x + z),
            0.2 * (6.0 * z + y + 2.0 * x),
        ],
        [
            0.001 * (x * x + 2.0 * x * y + y * z + 3.0 * x * z),
            0.001 * (2.0 * y * y + x * y + 2.0 * y * z + x * z),
            0.001 * (3.0 * z * z + x * y + y * z + 2.0 * x * z),
        ],
        [
            [
                0.001 * (2.0 * x + 2.0 * y + 3.0 * z),
                0.001 * (2.0 * x + z),
                0.001 * (y + 3.0 * x),
            ],
            [
                0.001 * (y + z),
                0.001 * (4.0 * y + x + 2.0 * z),
                0.001 * (2.0 * y + x),
            ],
            [
                0.001 * (y + 2.0 * z),
                0.001 * (x + z),
                0.001 * (6.0 * z + y + 2.0 * x),
            ],
        ],
    )
}
fn heat_source(x: f64, y: f64, z: f64, t: f64, law: TimeLaw) -> f64 {
    let theta = reference(x, y, z).0;
    1.3 * law.derivative(t) * theta + law.value(t) * (-0.7 * 2.4 + 0.4 * theta)
}
fn body_source(x: f64, y: f64, z: f64, t: f64, law: TimeLaw) -> Vec<f64> {
    let grad = reference(x, y, z).1;
    // Independent lambda=mu=40; lap(u)=(.002,.004,.006),
    // grad(div u)=(.005,.007,.011), and thermal stress coefficient=.2.
    [-0.48, -0.72, -1.12]
        .iter()
        .zip(grad)
        .map(|(b, g)| law.value(t) * (b + 0.2 * g))
        .collect()
}
fn combine(parts: &[(&TripletMatrix<f64>, f64)]) -> TripletMatrix<f64> {
    let first = parts[0].0;
    let mut out = TripletMatrix::new(first.num_rows, first.num_cols);
    for (a, scale) in parts {
        for k in 0..a.nnz() {
            out.push(a.rows[k], a.cols[k], scale * a.values[k]);
        }
    }
    out
}
fn thermal_system(
    matrix: TripletMatrix<f64>,
    rhs: Vec<f64>,
    mesh: &Mesh,
    space: &H1Space,
    t: f64,
    law: TimeLaw,
) -> Result<FemSystem<f64>, String> {
    let mut system = FemSystem::new(matrix, rhs);
    try_apply_dirichlet_values(
        &mut system,
        mesh,
        space,
        1,
        &|x, y, z| law.value(t) * reference(x, y, z).0,
        None,
    )
    .map_err(|e| e.to_string())?;
    Ok(system)
}
// Apply vector Dirichlet values with simultaneous column lifting. The scalar
// production helper accepts a single coordinate-only value for every vector
// component; this driver must prescribe distinct x/y/z displacement values.
fn mechanical_system(
    mut matrix: TripletMatrix<f64>,
    mut rhs: Vec<f64>,
    fixed: &[bool],
    coordinates: &[[f64; 3]],
    t: f64,
    law: TimeLaw,
) -> FemSystem<f64> {
    let prescribed: Vec<f64> = coordinates
        .iter()
        .enumerate()
        .map(|(d, p)| law.value(t) * reference(p[0], p[1], p[2]).2[d % 3])
        .collect();
    for k in 0..matrix.nnz() {
        let row = matrix.rows[k];
        let col = matrix.cols[k];
        if fixed[col] && !fixed[row] {
            rhs[row] -= matrix.values[k] * prescribed[col];
        }
        if fixed[row] || fixed[col] {
            matrix.values[k] = 0.0;
        }
    }
    for (d, &is_fixed) in fixed.iter().enumerate() {
        if is_fixed {
            matrix.push(d, d, 1.0);
            rhs[d] = prescribed[d];
        }
    }
    FemSystem::new(matrix, rhs)
}
struct ThermalLoad<'a> {
    points: &'a HashMap<[u64; 3], f64>,
}
impl LinearForm<f64> for ThermalLoad<'_> {
    fn integrate_point(&self, test: &BasisTabulation, weight: f64, local: &mut LocalVector<f64>) {
        ThermalStrainLoadForm {
            material: MATERIAL,
            temperature: MATERIAL.reference_temperature
                + self.points[&test.physical_coords.map(f64::to_bits)],
        }
        .integrate_point(test, weight, local);
    }
}
fn thermal_load(
    mesh: &Mesh,
    space: &H1Space,
    vector: &VectorH1Space,
    theta: &[f64],
    control: Control,
) -> Vec<f64> {
    if matches!(control, Control::NoThermalLoad) {
        return vec![0.0; vector.num_global_dofs(mesh)];
    }
    let mut points = HashMap::new();
    for element in 0..mesh.num_elements() {
        let dofs = space.local_to_global(mesh, element);
        for qp in QuadratureRule::new(mesh.elements[element].element_type, 4).iter() {
            let tab = space.tabulate(mesh, element, qp.xi(), qp.eta(), qp.zeta());
            let value = dofs
                .iter()
                .enumerate()
                .map(|(i, &d)| theta[d] * tab.value(i))
                .sum();
            points.insert(tab.physical_coords.map(f64::to_bits), value);
        }
    }
    assemble_linear(&ThermalLoad { points: &points }, vector, mesh, 4)
}
fn relative_free_residual(a: &CsrMatrix<f64>, x: &[f64], rhs: &[f64], fixed: &[bool]) -> f64 {
    let action = a.matvec(x);
    let error: f64 = (0..x.len())
        .filter(|&d| !fixed[d])
        .map(|d| (action[d] - rhs[d]).powi(2))
        .sum();
    let norm: f64 = (0..x.len())
        .filter(|&d| !fixed[d])
        .map(|d| rhs[d].powi(2))
        .sum();
    (error / norm).sqrt()
}
fn stress(gradient: [[f64; 3]; 3], theta: f64, exact: bool) -> [[f64; 3]; 3] {
    let (lambda, mu, thermal) = if exact {
        (40.0, 40.0, 0.2 * theta)
    } else {
        (
            MATERIAL.lame_lambda(),
            MATERIAL.shear_modulus(),
            (3.0 * MATERIAL.lame_lambda() + 2.0 * MATERIAL.shear_modulus())
                * MATERIAL.thermal_strain(MATERIAL.reference_temperature + theta),
        )
    };
    let trace: f64 = (0..3).map(|i| gradient[i][i]).sum();
    std::array::from_fn(|i| {
        std::array::from_fn(|j| {
            mu * (gradient[i][j] + gradient[j][i])
                + if i == j {
                    lambda * trace - thermal
                } else {
                    0.0
                }
        })
    })
}
#[derive(Debug)]
struct Metrics {
    temperature: f64,
    temperature_gradient: f64,
    displacement: f64,
    stress: f64,
    residual: f64,
    boundary: f64,
    heat_balance: f64,
    force_balance: f64,
    history_error: f64,
    history_change: f64,
    steps: usize,
    free: usize,
}
fn solve(
    n: usize,
    degree: PolynomialDegree,
    steps: usize,
    law: TimeLaw,
    control: Control,
) -> Result<Metrics, String> {
    let mut mesh = unit_cube_tetrahedra(n);
    for face in &mut mesh.boundaries {
        face.marker = 1;
    }
    let scalar = H1Space::new(degree);
    let vector = VectorH1Space::new(degree, 3);
    let nt = scalar.num_global_dofs(&mesh);
    let nu = vector.num_global_dofs(&mesh);
    let coords = scalar
        .dof_coordinates(&mesh)
        .ok_or("scalar coordinates missing")?;
    let ucoords = vector
        .dof_coordinates(&mesh)
        .ok_or("vector coordinates missing")?;
    let tboundary = scalar.boundary_dofs(&mesh, 1);
    let uboundary = vector.boundary_dofs(&mesh, 1);
    let mut tfixed = vec![false; nt];
    let mut ufixed = vec![false; nu];
    for &d in &tboundary {
        tfixed[d] = true;
    }
    for &d in &uboundary {
        ufixed[d] = true;
    }
    let free = (nt - tboundary.len()).min(nu - uboundary.len());
    if free == 0 {
        return Err("thermoelastic MMS has no free thermal/mechanical unknowns".into());
    }
    let capacity = if matches!(control, Control::NoCapacity) {
        0.0
    } else {
        1.3
    };
    let gamma = if matches!(control, Control::NoRelaxation) {
        0.0
    } else {
        0.4
    };
    let blocks = build_thermoelastic_coupled_blocks(
        &mesh,
        degree,
        3,
        IsotropicMaterial::new(100.0, 0.25),
        None,
        ThermalMaterial::new(0.7, 1.0, capacity),
        gamma,
    );
    let dt = 1.0 / steps as f64;
    let lhs = combine(&[
        (&blocks.heat_capacity, 1.0 / dt),
        (&blocks.heat_conduction, 0.5),
        (&blocks.damping_coupling, 0.5),
    ]);
    let raw = lhs.to_csr();
    let mass = blocks.heat_capacity.to_csr();
    let damping = blocks.damping_coupling.to_csr();
    let mechanical_raw = blocks.mechanical_stiffness.to_csr();
    let tfactor = SparseCholeskyFactorization::factorize(
        &thermal_system(lhs.clone(), vec![0.0; nt], &mesh, &scalar, 0.0, law)?
            .matrix
            .to_csr(),
    )
    .map_err(|e| e.to_string())?;
    let ufactor = SparseCholeskyFactorization::factorize(
        &mechanical_system(
            blocks.mechanical_stiffness.clone(),
            vec![0.0; nu],
            &ufixed,
            &ucoords,
            0.0,
            law,
        )
        .matrix
        .to_csr(),
    )
    .map_err(|e| e.to_string())?;
    let source = |time: f64| {
        if matches!(control, Control::NoHeatSource) {
            vec![0.0; nt]
        } else {
            assemble_linear(
                &HeatSourceForm {
                    source_fn: |x, y, z| heat_source(x, y, z, time, law),
                },
                &scalar,
                &mesh,
                4,
            )
        }
    };
    let mut theta: Vec<f64> = coords
        .iter()
        .map(|p| reference(p[0], p[1], p[2]).0)
        .collect();
    let initial = theta.clone();
    let mut previous_source = source(0.0);
    let mut displacement = vec![0.0; nu];
    let mut residual = 0.0_f64;
    let mut boundary = 0.0_f64;
    let mut heat_balance = 0.0_f64;
    let mut force_balance = 0.0_f64;
    let mut history_error = 0.0_f64;
    for step in 0..=steps {
        let time = step as f64 * dt;
        if step > 0 {
            let current_source = source(time);
            let old_mass = mass.matvec(&theta);
            let old_lhs = raw.matvec(&theta);
            let old = theta.clone();
            let rhs: Vec<f64> = (0..nt)
                .map(|i| {
                    2.0 * old_mass[i] / dt - old_lhs[i]
                        + 0.5 * (current_source[i] + previous_source[i])
                })
                .collect();
            theta = tfactor
                .solve(&thermal_system(lhs.clone(), rhs.clone(), &mesh, &scalar, time, law)?.rhs)
                .map_err(|e| e.to_string())?;
            residual = residual.max(relative_free_residual(&raw, &theta, &rhs, &tfixed));
            let action = raw.matvec(&theta);
            let change: Vec<_> = theta.iter().zip(&old).map(|(a, b)| (a - b) / dt).collect();
            let mean: Vec<_> = theta.iter().zip(&old).map(|(a, b)| 0.5 * (a + b)).collect();
            let stored: f64 = mass.matvec(&change).iter().sum();
            let relaxed: f64 = damping.matvec(&mean).iter().sum();
            let injected: f64 = current_source
                .iter()
                .zip(&previous_source)
                .map(|(a, b)| 0.5 * (a + b))
                .sum();
            let outward: f64 = -tboundary.iter().map(|&d| action[d] - rhs[d]).sum::<f64>();
            let scale = stored
                .abs()
                .max(relaxed.abs())
                .max(injected.abs())
                .max(outward.abs());
            heat_balance = heat_balance.max((stored + relaxed + outward - injected).abs() / scale);
            previous_source = current_source;
        }
        let body = assemble_linear(
            &BodyForceForm {
                force_fn: |x, y, z| body_source(x, y, z, time, law),
            },
            &vector,
            &mesh,
            4,
        );
        let thermal = thermal_load(&mesh, &scalar, &vector, &theta, control);
        let rhs: Vec<_> = body.iter().zip(thermal).map(|(a, b)| a + b).collect();
        displacement = ufactor
            .solve(
                &mechanical_system(
                    blocks.mechanical_stiffness.clone(),
                    rhs.clone(),
                    &ufixed,
                    &ucoords,
                    time,
                    law,
                )
                .rhs,
            )
            .map_err(|e| e.to_string())?;
        if theta.iter().chain(&displacement).any(|v| !v.is_finite()) {
            return Err("nonfinite thermoelastic history".into());
        }
        residual = residual.max(relative_free_residual(
            &mechanical_raw,
            &displacement,
            &rhs,
            &ufixed,
        ));
        let action = mechanical_raw.matvec(&displacement);
        for axis in 0..3 {
            let load: f64 = (axis..nu).step_by(3).map(|d| rhs[d]).sum();
            let reaction: f64 = uboundary
                .iter()
                .filter(|&&d| d % 3 == axis)
                .map(|&d| action[d] - rhs[d])
                .sum();
            force_balance = force_balance
                .max((load + reaction).abs() / rhs.iter().map(|v| v.abs()).sum::<f64>());
        }
        let mut terror = 0.0;
        let mut tnorm = 0.0;
        let mut uerror = 0.0;
        let mut unorm = 0.0;
        for (d, p) in coords.iter().enumerate() {
            let expected = law.value(time) * reference(p[0], p[1], p[2]).0;
            if tfixed[d] {
                boundary = boundary.max((theta[d] - expected).abs());
            } else {
                terror += (theta[d] - expected).powi(2);
                tnorm += expected.powi(2);
            }
        }
        for (d, p) in ucoords.iter().enumerate() {
            let expected = law.value(time) * reference(p[0], p[1], p[2]).2[d % 3];
            if ufixed[d] {
                boundary = boundary.max((displacement[d] - expected).abs());
            } else {
                uerror += (displacement[d] - expected).powi(2);
                unorm += expected.powi(2);
            }
        }
        history_error = history_error
            .max((terror / tnorm).sqrt())
            .max((uerror / unorm).sqrt());
    }
    let mut errors = [0.0; 4];
    let mut norms = [0.0; 4];
    let g = law.value(1.0);
    for element in 0..mesh.num_elements() {
        let tdofs = scalar.local_to_global(&mesh, element);
        let udofs = vector.local_to_global(&mesh, element);
        for qp in QuadratureRule::new(mesh.elements[element].element_type, 5).iter() {
            let tab = scalar.tabulate(&mesh, element, qp.xi(), qp.eta(), qp.zeta());
            let vtab = vector.tabulate(&mesh, element, qp.xi(), qp.eta(), qp.zeta());
            let [x, y, z] = tab.physical_coords;
            let (t, dt, u, du) = reference(x, y, z);
            let w = qp.weight * tab.det_j;
            let actual_t: f64 = tdofs
                .iter()
                .enumerate()
                .map(|(i, &d)| theta[d] * tab.value(i))
                .sum();
            let actual_dt: [f64; 3] = std::array::from_fn(|a| {
                tdofs
                    .iter()
                    .enumerate()
                    .map(|(i, &d)| theta[d] * tab.grad(i, a))
                    .sum()
            });
            let actual_u: [f64; 3] = std::array::from_fn(|a| {
                udofs
                    .iter()
                    .enumerate()
                    .map(|(i, &d)| displacement[d] * vtab.vector_value(i, a))
                    .sum()
            });
            let actual_du: [[f64; 3]; 3] = std::array::from_fn(|a| {
                std::array::from_fn(|b| {
                    udofs
                        .iter()
                        .enumerate()
                        .filter(|(i, _)| i % 3 == a)
                        .map(|(i, &d)| displacement[d] * vtab.grad(i, b))
                        .sum()
                })
            });
            let actual_stress = stress(actual_du, actual_t, false);
            let expected_stress = stress(du.map(|r| r.map(|v| g * v)), g * t, true);
            errors[0] += w * (actual_t - g * t).powi(2);
            norms[0] += w * (g * t).powi(2);
            for a in 0..3 {
                errors[1] += w * (actual_dt[a] - g * dt[a]).powi(2);
                norms[1] += w * (g * dt[a]).powi(2);
                errors[2] += w * (actual_u[a] - g * u[a]).powi(2);
                norms[2] += w * (g * u[a]).powi(2);
                for b in 0..3 {
                    errors[3] += w * (actual_stress[a][b] - expected_stress[a][b]).powi(2);
                    norms[3] += w * expected_stress[a][b].powi(2);
                }
            }
        }
    }
    let history_change = theta
        .iter()
        .zip(&initial)
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f64>()
        .sqrt()
        / initial.iter().map(|v| v * v).sum::<f64>().sqrt();
    Ok(Metrics {
        temperature: (errors[0] / norms[0]).sqrt(),
        temperature_gradient: (errors[1] / norms[1]).sqrt(),
        displacement: (errors[2] / norms[2]).sqrt(),
        stress: (errors[3] / norms[3]).sqrt(),
        residual,
        boundary,
        heat_balance,
        force_balance,
        history_error,
        history_change,
        steps,
        free,
    })
}
fn run() -> Result<QaResult, String> {
    let mut spatial = Vec::new();
    for n in [4, 8, 16] {
        let m = solve(n, PolynomialDegree::P1, 8, TimeLaw::Linear, Control::Full)?;
        println!("thermoelastic P1 n={n}: {m:?}");
        spatial.push(m);
    }
    let mut temporal = Vec::new();
    for steps in [4, 8, 16] {
        let m = solve(
            2,
            PolynomialDegree::P2,
            steps,
            TimeLaw::Exponential,
            Control::Full,
        )?;
        println!("thermoelastic P2 time {steps}: {m:?}");
        temporal.push(m);
    }
    let patch = solve(2, PolynomialDegree::P2, 8, TimeLaw::Linear, Control::Full)?;
    println!("thermoelastic exact P2: {patch:?}");
    let mut negative = f64::INFINITY;
    for control in [
        Control::NoCapacity,
        Control::NoRelaxation,
        Control::NoThermalLoad,
        Control::NoHeatSource,
    ] {
        let m = solve(2, PolynomialDegree::P2, 8, TimeLaw::Linear, control)?;
        println!("thermoelastic {control:?}: {m:?}");
        negative = negative.min(m.history_error);
    }
    let value_rate = |select: fn(&Metrics) -> f64, levels: &[Metrics]| {
        levels
            .windows(2)
            .map(|m| (select(&m[0]) / select(&m[1])).log2())
            .fold(f64::INFINITY, f64::min)
    };
    let all: Vec<_> = spatial.iter().chain(&temporal).chain([&patch]).collect();
    let mut result = QaResult::passed(CASE)
        .requested_backend("cpu")
        .actual_backend("cpu");
    for (name, value, bound) in [
        ("fine_temperature_error", spatial[2].temperature, 0.005),
        ("fine_displacement_error", spatial[2].displacement, 0.005),
        (
            "fine_temperature_gradient_error",
            spatial[2].temperature_gradient,
            0.1,
        ),
        ("fine_stress_error", spatial[2].stress, 0.1),
        (
            "fine_temporal_field_error",
            temporal[2]
                .temperature
                .max(temporal[2].displacement)
                .max(temporal[2].stress),
            0.001,
        ),
        (
            "exact_p2_error",
            patch
                .temperature
                .max(patch.displacement)
                .max(patch.stress)
                .max(patch.temperature_gradient),
            1e-9,
        ),
        (
            "physical_residual",
            all.iter().map(|m| m.residual).fold(0.0_f64, f64::max),
            1e-9,
        ),
        (
            "heat_balance",
            all.iter().map(|m| m.heat_balance).fold(0.0_f64, f64::max),
            1e-9,
        ),
        (
            "force_balance",
            all.iter().map(|m| m.force_balance).fold(0.0_f64, f64::max),
            1e-9,
        ),
        (
            "boundary_error",
            all.iter().map(|m| m.boundary).fold(0.0_f64, f64::max),
            1e-10,
        ),
    ] {
        if !value.is_finite() || value > bound {
            return Err(format!("{name}={value:.6e} exceeds {bound}"));
        }
        result = result.metric(name, value).maximum(name, bound);
    }
    for (name, value, bound) in [
        (
            "spatial_temperature_rate",
            value_rate(|m| m.temperature, &spatial),
            1.8,
        ),
        (
            "spatial_displacement_rate",
            value_rate(|m| m.displacement, &spatial),
            1.8,
        ),
        (
            "spatial_stress_rate",
            value_rate(|m| m.stress, &spatial),
            0.9,
        ),
        (
            "spatial_temperature_gradient_rate",
            value_rate(|m| m.temperature_gradient, &spatial),
            0.9,
        ),
        (
            "temporal_temperature_rate",
            value_rate(|m| m.temperature, &temporal),
            1.8,
        ),
        (
            "temporal_displacement_rate",
            value_rate(|m| m.displacement, &temporal),
            1.8,
        ),
        (
            "temporal_stress_rate",
            value_rate(|m| m.stress, &temporal),
            1.8,
        ),
        (
            "minimum_history_change",
            all.iter()
                .map(|m| m.history_change)
                .fold(f64::INFINITY, f64::min),
            0.1,
        ),
        ("minimum_omitted_term_error", negative, 1e-3),
    ] {
        if !value.is_finite() || value < bound {
            return Err(format!("{name}={value:.6e} below {bound}"));
        }
        result = result.metric(name, value).minimum(name, bound);
    }
    Ok(result.metric("executed_time_steps",all.iter().map(|m|m.steps).sum::<usize>() as f64).minimum("executed_time_steps",60.0)
        .metric("minimum_free_field_unknowns",all.iter().map(|m|m.free).min().unwrap_or(0) as f64).minimum("minimum_free_field_unknowns",1.0)
        .metric("negative_controls",4.0).minimum("negative_controls",4.0)
        .reference("independent quadratic temperature and three displacement fields, constant-Lame thermoelastic strong-form sources and stress; actual CN history and FE thermal load")
        .reference_checksum("analytic:3d-transient-thermal-relaxation-quasistatic-thermoelasticity:v1")
        .input_checksum("P1n4-8-16:linear8:P2n2:exp4-8-16:linear8:E100:nu.25:alpha.001:C1.3:k.7:gamma.4:four-controls")
        .dimension("3D tetrahedral cube, temperature and three displacement components with mixed xyz curvature; all fields have free interior unknowns")
        .formulation("native thermoelastic coupled blocks, CN heat capacity/conduction/scalar-relaxation history and quasi-static mechanical solve each step using solved-temperature thermal strain")
        .tolerance("value spatial/time rates>=1.8; stress spatial/time rates>=0.9/1.8; P2/residual/heat/force balance<=1e-9; four history controls>=1e-3"))
}
fn main() {
    sonium_qa::fem_physics_3d::run_bilinear_case(
        "fem.thermoelasticity-3d.v1",
        &ThermoelasticDampingForm {
            coupling_coefficient: 1.0,
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
