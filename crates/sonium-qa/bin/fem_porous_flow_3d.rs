//! Darcy form contract and solved transient 3D anisotropic pore-pressure/flux convergence.

use sonium_fem::assembly::{TripletMatrix, assemble_bilinear, assemble_linear};
use sonium_fem::basis::PolynomialDegree;
use sonium_fem::boundary::generic::{FemSystem, try_apply_dirichlet_values};
use sonium_fem::forms::{BilinearForm, LocalMatrix};
use sonium_fem::mesh::unit_cube_tetrahedra;
use sonium_fem::physics::fluid::porous_flow::{
    DarcyMaterial, DarcyPressureForm, DarcySourceForm, DarcyStorageForm,
};
use sonium_fem::quadrature::QuadratureRule;
use sonium_fem::spaces::{BasisTabulation, FunctionSpace, H1Space};
use sonium_qa::{QaResult, QaStatus};
use sonium_solvers::SparseCholeskyFactorization;

const CASE_ID: &str = "fem.porous-flow-3d-transient-darcy.v1";
const FINAL_TIME: f64 = 0.5;
// k/mu gives the full SPD mobility tensor below. Nonunit viscosity catches
// assembly that accidentally uses permeability in place of hydraulic mobility.
const MATERIAL: DarcyMaterial = DarcyMaterial {
    permeability: [[1.2, 0.2, 0.1], [0.2, 1.8, 0.3], [0.1, 0.3, 2.4]],
    viscosity: 2.0,
    specific_storage: 1.3,
    porosity: 0.35,
};
const REFERENCE_MOBILITY: [[f64; 3]; 3] = [[0.6, 0.1, 0.05], [0.1, 0.9, 0.15], [0.05, 0.15, 1.2]];

fn shape(x: f64, y: f64, z: f64) -> (f64, [f64; 3]) {
    (
        1.0 + x * x + 2.0 * y * y + 3.0 * z * z + x * y + 2.0 * y * z + 3.0 * x * z,
        [
            2.0 * x + y + 3.0 * z,
            4.0 * y + x + 2.0 * z,
            6.0 * z + 2.0 * y + 3.0 * x,
        ],
    )
}

#[derive(Debug, Clone, Copy)]
enum TimeLaw {
    Linear,
    Exponential,
}
impl TimeLaw {
    fn value(self, t: f64) -> f64 {
        match self {
            Self::Linear => 1.0 + 0.6 * t,
            Self::Exponential => (-t).exp(),
        }
    }
    fn derivative(self, t: f64) -> f64 {
        match self {
            Self::Linear => 0.6,
            Self::Exponential => -(-t).exp(),
        }
    }
}

// Independent strong-equation forcing, never computed as K times a reference
// vector. The full Hessian contraction includes all off-diagonal diffusivities.
fn source(x: f64, y: f64, z: f64, t: f64, law: TimeLaw) -> f64 {
    let (q, _) = shape(x, y, z);
    const HESSIAN: [[f64; 3]; 3] = [[2.0, 1.0, 3.0], [1.0, 4.0, 2.0], [3.0, 2.0, 6.0]];
    let diffusion: f64 = (0..3)
        .flat_map(|i| (0..3).map(move |j| REFERENCE_MOBILITY[i][j] * HESSIAN[i][j]))
        .sum();
    1.3 * law.derivative(t) * q - law.value(t) * diffusion
}

#[derive(Debug, Clone, Copy)]
struct StepForm {
    material: DarcyMaterial,
    inverse_dt: f64,
}
impl BilinearForm<f64> for StepForm {
    fn integrate_point(
        &self,
        test: &BasisTabulation,
        trial: &BasisTabulation,
        weight: f64,
        local: &mut LocalMatrix<f64>,
    ) {
        DarcyStorageForm {
            specific_storage: self.material.specific_storage * self.inverse_dt,
        }
        .integrate_point(test, trial, weight, local);
        let mut spatial = LocalMatrix::zeros(test.n_dofs);
        DarcyPressureForm::new(&self.material).integrate_point(test, trial, weight, &mut spatial);
        for i in 0..test.n_dofs {
            for j in 0..trial.n_dofs {
                local.add(i, j, 0.5 * spatial.get(i, j));
            }
        }
    }
}

#[derive(Debug)]
struct Metrics {
    value_error: f64,
    flux_error: f64,
    residual: f64,
    boundary: f64,
    free_dofs: usize,
    steps: usize,
    history_change: f64,
    maximum_history_error: f64,
    mass_balance: f64,
}

fn solve(
    n: usize,
    degree: PolynomialDegree,
    steps: usize,
    law: TimeLaw,
    material: DarcyMaterial,
) -> Result<Metrics, String> {
    let mut mesh = unit_cube_tetrahedra(n);
    for face in &mut mesh.boundaries {
        face.marker = 1;
    }
    let space = H1Space::new(degree);
    let coordinates = space
        .dof_coordinates(&mesh)
        .ok_or("Darcy H1 has no coordinates")?;
    let count = space.num_global_dofs(&mesh);
    let boundary = space.boundary_dofs(&mesh, 1);
    let mut constrained = vec![false; count];
    for &i in &boundary {
        constrained[i] = true;
    }
    let free_dofs = count - boundary.len();
    if free_dofs == 0 {
        return Err("Darcy MMS has no free unknowns".into());
    }
    let dt = FINAL_TIME / steps as f64;
    // Fourth-order quadrature covers the P2 mass and polynomial forcing.
    let lhs: TripletMatrix<f64> = assemble_bilinear(
        &StepForm {
            material,
            inverse_dt: 1.0 / dt,
        },
        &space,
        &space,
        &mesh,
        4,
    );
    let raw = lhs.to_csr();
    let mass =
        assemble_bilinear(&DarcyStorageForm::new(&material), &space, &space, &mesh, 4).to_csr();
    let mut factor_system = FemSystem::new(lhs.clone(), vec![0.0; count]);
    try_apply_dirichlet_values(&mut factor_system, &mesh, &space, 1, &|_, _, _| 0.0, None)
        .map_err(|e| e.to_string())?;
    let factor = SparseCholeskyFactorization::factorize(&factor_system.matrix.to_csr())
        .map_err(|e| e.to_string())?;
    let mut values: Vec<f64> = coordinates
        .iter()
        .map(|p| shape(p[0], p[1], p[2]).0)
        .collect();
    let initial = values.clone();
    let mut previous_source = assemble_linear(
        &DarcySourceForm {
            source_fn: |x, y, z| source(x, y, z, 0.0, law),
        },
        &space,
        &mesh,
        4,
    );
    let mut residual = 0.0_f64;
    let mut boundary_error = 0.0_f64;
    let mut maximum_history_error = 0.0_f64;
    let mut mass_balance = 0.0_f64;
    for step in 1..=steps {
        let time = step as f64 * dt;
        let current_source = assemble_linear(
            &DarcySourceForm {
                source_fn: |x, y, z| source(x, y, z, time, law),
            },
            &space,
            &mesh,
            4,
        );
        let old_mass = mass.matvec(&values);
        let old_lhs = raw.matvec(&values);
        // CN history: (M/dt-A/2)u_old + (f_new+f_old)/2.
        let physical_rhs: Vec<f64> = (0..count)
            .map(|i| {
                2.0 * old_mass[i] / dt - old_lhs[i] + 0.5 * (current_source[i] + previous_source[i])
            })
            .collect();
        let mut system = FemSystem::new(lhs.clone(), physical_rhs.clone());
        try_apply_dirichlet_values(
            &mut system,
            &mesh,
            &space,
            1,
            &|x, y, z| law.value(time) * shape(x, y, z).0,
            None,
        )
        .map_err(|e| e.to_string())?;
        let old_values = values.clone();
        values = factor.solve(&system.rhs).map_err(|e| e.to_string())?;
        if values.iter().any(|v| !v.is_finite()) {
            return Err("nonfinite Darcy state".into());
        }
        let action = raw.matvec(&values);
        let numerator: f64 = (0..count)
            .filter(|&i| !constrained[i])
            .map(|i| (action[i] - physical_rhs[i]).powi(2))
            .sum();
        let denominator: f64 = (0..count)
            .filter(|&i| !constrained[i])
            .map(|i| physical_rhs[i].powi(2))
            .sum();
        if denominator <= 0.0 {
            return Err("empty transient Darcy history/source".into());
        }
        residual = residual.max((numerator / denominator).sqrt());
        for &i in &boundary {
            let p = coordinates[i];
            boundary_error =
                boundary_error.max((values[i] - law.value(time) * shape(p[0], p[1], p[2]).0).abs());
        }
        let mut history_error = 0.0;
        let mut history_norm = 0.0;
        for (i, p) in coordinates
            .iter()
            .enumerate()
            .filter(|(i, _)| !constrained[*i])
        {
            let expected = law.value(time) * shape(p[0], p[1], p[2]).0;
            history_error += (values[i] - expected).powi(2);
            history_norm += expected.powi(2);
        }
        maximum_history_error = maximum_history_error.max((history_error / history_norm).sqrt());
        // The original weak boundary reactions equal negative outward Darcy
        // flux. Include storage change and the time-averaged volumetric source.
        let change: Vec<f64> = values
            .iter()
            .zip(&old_values)
            .map(|(a, b)| (a - b) / dt)
            .collect();
        let stored = mass.matvec(&change).iter().sum::<f64>();
        let injected = current_source
            .iter()
            .zip(&previous_source)
            .map(|(a, b)| 0.5 * (a + b))
            .sum::<f64>();
        let outward = -boundary
            .iter()
            .map(|&i| action[i] - physical_rhs[i])
            .sum::<f64>();
        let scale = injected.abs().max(stored.abs()).max(outward.abs());
        if scale == 0.0 {
            return Err("empty Darcy mass budget".into());
        }
        mass_balance = mass_balance.max((stored + outward - injected).abs() / scale);
        previous_source = current_source;
    }
    let mut value_error = 0.0;
    let mut value_norm = 0.0;
    let mut flux_error = 0.0;
    let mut flux_norm = 0.0;
    for element in 0..mesh.num_elements() {
        let dofs = space.local_to_global(&mesh, element);
        for qp in QuadratureRule::new(mesh.elements[element].element_type, 5).iter() {
            let tab = space.tabulate(&mesh, element, qp.xi(), qp.eta(), qp.zeta());
            let p = tab.physical_coords;
            let (q, g) = shape(p[0], p[1], p[2]);
            let expected = law.value(FINAL_TIME) * q;
            let actual: f64 = dofs
                .iter()
                .enumerate()
                .map(|(i, &d)| values[d] * tab.value(i))
                .sum();
            let weight = qp.weight * tab.det_j;
            value_error += weight * (actual - expected).powi(2);
            value_norm += weight * expected.powi(2);
            let actual_gradient: [f64; 3] = std::array::from_fn(|axis| {
                dofs.iter()
                    .enumerate()
                    .map(|(i, &d)| values[d] * tab.grad(i, axis))
                    .sum()
            });
            let mobility = material.hydraulic_mobility();
            for axis in 0..3 {
                let actual_flux: f64 = (0..3)
                    .map(|j| -mobility[axis][j] * actual_gradient[j])
                    .sum();
                let expected_flux: f64 = (0..3)
                    .map(|j| -REFERENCE_MOBILITY[axis][j] * law.value(FINAL_TIME) * g[j])
                    .sum();
                flux_error += weight * (actual_flux - expected_flux).powi(2);
                flux_norm += weight * expected_flux.powi(2);
            }
        }
    }
    let history_change = values
        .iter()
        .zip(&initial)
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f64>()
        .sqrt()
        / initial.iter().map(|v| v * v).sum::<f64>().sqrt();
    Ok(Metrics {
        value_error: (value_error / value_norm).sqrt(),
        flux_error: (flux_error / flux_norm).sqrt(),
        residual,
        boundary: boundary_error,
        free_dofs,
        steps,
        history_change,
        maximum_history_error,
        mass_balance,
    })
}

fn run() -> Result<QaResult, String> {
    let mut spatial = Vec::new();
    for n in [6, 12, 24] {
        let metrics = solve(n, PolynomialDegree::P1, 12, TimeLaw::Linear, MATERIAL)?;
        println!("Darcy spatial n={n}: {metrics:?}");
        spatial.push(metrics);
    }
    let mut temporal = Vec::new();
    for steps in [4, 8, 16] {
        let metrics = solve(
            2,
            PolynomialDegree::P2,
            steps,
            TimeLaw::Exponential,
            MATERIAL,
        )?;
        println!("Darcy temporal steps={steps}: {metrics:?}");
        temporal.push(metrics);
    }
    for levels in [&spatial, &temporal] {
        if !levels.windows(2).all(|pair| {
            pair[0].value_error > pair[1].value_error && pair[0].flux_error > pair[1].flux_error
        }) {
            return Err("Darcy space/time value and gradient errors do not contract".into());
        }
    }
    let value_rate = (spatial[1].value_error / spatial[2].value_error).log2();
    let flux_rate = (spatial[1].flux_error / spatial[2].flux_error).log2();
    let time_rate = (temporal[1].value_error / temporal[2].value_error).log2();
    let time_flux_rate = (temporal[1].flux_error / temporal[2].flux_error).log2();
    let residual = spatial
        .iter()
        .chain(&temporal)
        .map(|m| m.residual)
        .fold(0.0_f64, f64::max);
    let boundary = spatial
        .iter()
        .chain(&temporal)
        .map(|m| m.boundary)
        .fold(0.0_f64, f64::max);
    let free_dofs = spatial
        .iter()
        .chain(&temporal)
        .map(|m| m.free_dofs)
        .min()
        .ok_or("no Darcy cases")?;
    let history_change = spatial
        .iter()
        .chain(&temporal)
        .map(|m| m.history_change)
        .fold(f64::INFINITY, f64::min);
    // Independent source stays unchanged in all negative controls. Measure
    // every evolved interior state so even-step CN cancellation cannot hide
    // omitted storage. Tensor controls detect missing cross or z mobility.
    let mut ablation_error = f64::INFINITY;
    for material in [
        DarcyMaterial {
            specific_storage: 0.0,
            ..MATERIAL
        },
        DarcyMaterial {
            permeability: [[1.2, 0.0, 0.0], [0.0, 1.8, 0.0], [0.0, 0.0, 2.4]],
            ..MATERIAL
        },
        DarcyMaterial {
            permeability: [[1.2, 0.2, 0.0], [0.2, 1.8, 0.0], [0.0, 0.0, 0.4]],
            ..MATERIAL
        },
    ] {
        let control = solve(2, PolynomialDegree::P2, 12, TimeLaw::Linear, material)?;
        println!("Darcy omitted-term control: {control:?}");
        ablation_error = ablation_error.min(control.maximum_history_error);
    }
    let mass_balance = spatial
        .iter()
        .chain(&temporal)
        .map(|m| m.mass_balance)
        .fold(0.0_f64, f64::max);
    let mut result = QaResult::passed(CASE_ID)
        .requested_backend("cpu")
        .actual_backend("cpu");
    for (name, value, bound) in [
        ("fine_spatial_value_error", spatial[2].value_error, 1e-3),
        ("fine_spatial_flux_error", spatial[2].flux_error, 0.06),
        ("fine_temporal_value_error", temporal[2].value_error, 1e-4),
        ("fine_temporal_flux_error", temporal[2].flux_error, 1e-3),
        ("physical_residual", residual, 1e-9),
        ("mass_balance_error", mass_balance, 1e-9),
        ("boundary_error", boundary, 1e-10),
    ] {
        if !value.is_finite() || value > bound {
            return Err(format!("{name} {value:.3e} > {bound:.3e}"));
        }
        result = result.metric(name, value).maximum(name, bound);
    }
    for (name, value, bound) in [
        ("spatial_value_rate", value_rate, 1.8),
        ("spatial_flux_rate", flux_rate, 0.9),
        ("temporal_value_rate", time_rate, 1.8),
        ("temporal_flux_rate", time_flux_rate, 1.8),
        ("history_change", history_change, 0.1),
        ("minimum_omitted_term_error", ablation_error, 1e-3),
    ] {
        if !value.is_finite() || value < bound {
            return Err(format!("{name} {value:.3e} < {bound:.3e}"));
        }
        result = result.metric(name, value).minimum(name, bound);
    }
    Ok(result.metric("minimum_free_dofs",free_dofs as f64).minimum("minimum_free_dofs",1.0)
        .metric("executed_time_steps",spatial.iter().chain(&temporal).map(|m|m.steps).sum::<usize>() as f64)
        .minimum("executed_time_steps",64.0)
        .metric("negative_controls",3.0).minimum("negative_controls",3.0)
        .seed("anisotropic-Darcy:k-mu2-storage1.3:P1n6-12-24:linear:P2n2:exp4-8-16")
        .input_checksum("native-Darcy-pressure-storage-source:CN:3-controls:v1")
        .reference("independent S*p_t-div(M grad p)=f; p=g(t)*(1+x^2+2y^2+3z^2+xy+2yz+3xz); Darcy flux=-M grad p")
        .reference_checksum("analytic:3d-anisotropic-transient-Darcy:v1")
        .dimension("3D unit cube, mixed xyz Hessian and full permeability tensor, evolving boundary data and free pressure unknowns")
        .formulation("production Darcy pressure/storage/source forms; CN actual history, sparse Cholesky; independently recovered Darcy flux and weak-boundary mass budget")
        .tolerance("pressure spatial/time rates>=1.8; flux spatial/time rates>=0.9/1.8; residual and mass budget<=1e-9; three omitted-term history controls"))
}

fn main() {
    sonium_qa::fem_physics_3d::run_bilinear_case(
        "fem.porous-flow-3d.v1",
        &DarcyPressureForm {
            mobility: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        },
        true,
    );
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
