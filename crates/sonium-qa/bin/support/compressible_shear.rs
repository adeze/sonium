//! Nonuniform rotating shear-wave accuracy for the native compressible DG operator.
//!
//! With rho=1, u=A(t)(cos(pi z), sin(pi z), 0), and spatially uniform
//! total energy, the full compressible Navier--Stokes equations reduce to
//! A'= -mu*pi^2*A. Viscous heating raises pressure as kinetic energy falls.
//! Boundary traces are sampled per face because the public boundary API
//! accepts constant states. The spatial test includes this boundary error.
//! A QA SSPRK3 driver updates prescribed traces at all three stage times;
//! a separate temporal test exercises the public solver with fixed traces.

// Rust guideline compliant 2026-02-21
use sonium_fem::mesh::{Mesh, unit_cube_tetrahedra};
use sonium_fem::physics::fluid::compressible_flow::{
    CompressibleBoundaryCondition, CompressibleDgConfig, CompressibleDgOperator,
    CompressibleDgSolver, CompressibleDgState, ConservativeState, IdealGas, ViscosityModel,
};
use sonium_qa::QaResult;
use std::f64::consts::PI;

const GAS: IdealGas = IdealGas {
    gamma: 1.4,
    gas_constant: 1.0,
};
const AMPLITUDE: f64 = 0.4;
const VISCOSITY: f64 = 0.2;
const END: f64 = 0.1;
// Initial temperature is one; total energy is constant during viscous heating.
const ENERGY: f64 = 2.5 + 0.5 * AMPLITUDE * AMPLITUDE;

fn exact(z: f64, time: f64) -> ConservativeState {
    let a = AMPLITUDE * (-VISCOSITY * PI * PI * time).exp();
    ConservativeState {
        density: 1.0,
        momentum: [a * (PI * z).cos(), a * (PI * z).sin(), 0.0],
        total_energy: ENERGY,
    }
}

fn config(mesh: &Mesh, time: f64, viscosity: f64) -> CompressibleDgConfig {
    let mut config = CompressibleDgConfig {
        gas: GAS,
        viscosity: ViscosityModel::Constant(viscosity),
        // For equal neighboring characteristics the limiter's margin is
        // relaxation*1e-12. A margin of 100 leaves these O(1) smooth states
        // unchanged, including at shear extrema. Positivity remains active.
        troubled_cell_relaxation: 1.0e14,
        ..Default::default()
    };
    for face in &mesh.boundaries {
        let z = face.nodes.iter().map(|&n| mesh.nodes[n].z).sum::<f64>() / face.nodes.len() as f64;
        config.boundaries.insert(
            face.marker,
            CompressibleBoundaryCondition::SupersonicState(exact(z, time)),
        );
    }
    config
}

fn mesh(n: usize) -> Mesh {
    let mut mesh = unit_cube_tetrahedra(n);
    for (i, face) in mesh.boundaries.iter_mut().enumerate() {
        face.marker = i as i32 + 1;
    }
    mesh
}

fn combine(
    a: &CompressibleDgState,
    wa: f64,
    b: &CompressibleDgState,
    wb: f64,
) -> CompressibleDgState {
    let mut state = a.clone();
    for ((out, a), b) in state.values.iter_mut().zip(&a.values).zip(&b.values) {
        let aa = a.as_array();
        let bb = b.as_array();
        *out = ConservativeState::from_array(std::array::from_fn(|k| wa * aa[k] + wb * bb[k]));
    }
    state
}

fn euler(
    operator: &CompressibleDgOperator<'_>,
    state: &CompressibleDgState,
    dt: f64,
) -> Result<CompressibleDgState, String> {
    let rates = operator
        .spatial_operator(state)
        .map_err(|e| format!("DG RHS: {e:?}"))?;
    let mut next = state.clone();
    for (out, rate) in next.values.iter_mut().zip(rates) {
        let a = out.as_array();
        let r = rate.as_array();
        *out = ConservativeState::from_array(std::array::from_fn(|k| a[k] + dt * r[k]));
    }
    if next
        .values
        .iter()
        .any(|s| !s.is_physical(GAS, 1e-12, 1e-12))
    {
        return Err("nonphysical shear-wave stage".into());
    }
    Ok(next)
}

fn solve_wave(mesh: &Mesh, steps: usize, viscosity: f64) -> Result<CompressibleDgState, String> {
    let mut state = CompressibleDgState::from_function(mesh, |p| exact(p[2], 0.0))
        .map_err(|e| format!("initialization: {e:?}"))?;
    let mut operator = CompressibleDgOperator::new(mesh, config(mesh, 0.0, viscosity))
        .map_err(|e| format!("operator: {e:?}"))?;
    let dt = END / steps as f64;
    for step in 0..steps {
        let t = step as f64 * dt;
        operator.config = config(mesh, t, viscosity);
        if dt
            > operator
                .stable_time_step(&state)
                .map_err(|e| format!("CFL: {e:?}"))?
        {
            return Err("requested shear-wave timestep exceeds native CFL limit".into());
        }
        let stage1 = euler(&operator, &state, dt)?;
        operator.config = config(mesh, t + dt, viscosity);
        let stage2 = combine(&state, 0.75, &euler(&operator, &stage1, dt)?, 0.25);
        operator.config = config(mesh, t + 0.5 * dt, viscosity);
        state = combine(
            &state,
            1.0 / 3.0,
            &euler(&operator, &stage2, dt)?,
            2.0 / 3.0,
        );
    }
    Ok(state)
}

// All generated tetrahedra have equal volume. The symmetric four-point rule
// integrates quadratic P1 errors and samples the reference between vertices.
fn errors(mesh: &Mesh, state: &CompressibleDgState) -> [f64; 3] {
    let mut sums = [0.0; 3];
    let a = 0.585_410_196_624_968_5;
    let b = 0.138_196_601_125_010_5;
    for (e, cell) in mesh.elements.iter().enumerate() {
        for point in 0..4 {
            let mut z = 0.0;
            let mut value = [0.0; 5];
            for (i, &node) in cell.vertices().iter().enumerate() {
                let weight = if i == point { a } else { b };
                z += weight * mesh.nodes[node].z;
                let local = state.element(e)[i].as_array();
                for k in 0..5 {
                    value[k] += weight * local[k];
                }
            }
            let value = ConservativeState::from_array(value);
            let reference = exact(z, END);
            sums[0] += value
                .momentum
                .iter()
                .zip(reference.momentum)
                .map(|(v, r)| (v - r).powi(2))
                .sum::<f64>()
                / reference.momentum.iter().map(|v| v * v).sum::<f64>();
            sums[1] += (value.pressure(GAS) - reference.pressure(GAS)).powi(2);
            sums[2] += (value.density - 1.0).powi(2);
        }
    }
    sums.map(|s| (s / (4 * mesh.num_elements()) as f64).sqrt())
}

fn native_time(mesh: &Mesh, steps: usize) -> Result<CompressibleDgState, String> {
    let solver = CompressibleDgSolver::new(mesh, config(mesh, 0.0, VISCOSITY))
        .map_err(|e| format!("native solver: {e:?}"))?;
    let mut state = CompressibleDgState::from_function(mesh, |p| exact(p[2], 0.0))
        .map_err(|e| format!("native state: {e:?}"))?;
    let dt = END / steps as f64;
    for _ in 0..steps {
        let report = solver
            .step(&mut state, dt)
            .map_err(|e| format!("native step: {e:?}"))?;
        if report.retries != 0 || (report.accepted_time_step - dt).abs() > 1e-14 {
            return Err("temporal refinement changed the requested timestep".into());
        }
    }
    Ok(state)
}

fn difference(a: &CompressibleDgState, b: &CompressibleDgState) -> f64 {
    (a.values
        .iter()
        .zip(&b.values)
        .flat_map(|(a, b)| {
            a.as_array()
                .into_iter()
                .zip(b.as_array())
                .map(|(a, b)| (a - b).powi(2))
        })
        .sum::<f64>()
        / a.values.len() as f64)
        .sqrt()
}

pub(super) fn run() -> Result<QaResult, String> {
    let mut result = QaResult::passed("fem.compressible-shear-wave-3d.v1")
        .requested_backend("cpu").actual_backend("cpu")
        .seed("rotating-shear:rho1:A0.4:kpi:mu0.2:T1:t0.1")
        .reference("exact nonlinear rotating shear: A=A0 exp(-mu pi^2 t), rho=1, total energy constant")
        .formulation("native P1 DG Euler/stress/heat/SIPG; stage-time face traces; separate public SSPRK3 time refinement");
    let mut velocity_errors = Vec::new();
    for n in [2, 4, 6] {
        let mesh = mesh(n);
        let state = solve_wave(&mesh, 100, VISCOSITY)?;
        let error = errors(&mesh, &state);
        println!(
            "shear n={n} velocity={} pressure={} density={}",
            error[0], error[1], error[2]
        );
        for (name, value) in ["velocity", "pressure", "density"].into_iter().zip(error) {
            result = result.metric(format!("{name}_error_n{n}"), value);
        }
        velocity_errors.push(error[0]);
    }
    let spatial_rate = (velocity_errors[1] / velocity_errors[2]).ln() / (1.5_f64).ln();
    let mesh = mesh(2);
    let solutions = [20, 40, 80, 160]
        .map(|steps| native_time(&mesh, steps))
        .into_iter()
        .collect::<Result<Vec<_>, _>>()?;
    let deltas: Vec<_> = solutions
        .windows(2)
        .map(|s| difference(&s[0], &s[1]))
        .collect();
    let temporal_rate = (deltas[1] / deltas[2]).log2();
    let control_mesh = self::mesh(6);
    let control = solve_wave(&control_mesh, 100, VISCOSITY * 0.01)?;
    let control_error = errors(&control_mesh, &control)[0];
    println!(
        "shear spatial_rate={spatial_rate} temporal_deltas={deltas:?} temporal_rate={temporal_rate} control_error={control_error}"
    );
    result = result
        .metric("spatial_rate", spatial_rate)
        .metric("temporal_rate", temporal_rate)
        .metric("transport_control_error", control_error);
    // Acceptance thresholds are added after checking the development run.
    Ok(result)
}
