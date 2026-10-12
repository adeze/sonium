//! Solved open-circuit piezoelectric thickness modes with field refinement.

#[expect(
    dead_code,
    reason = "Shared rod helper contains static and modal APIs; each binary exercises one."
)]
#[path = "support/piezo_bar.rs"]
mod piezo_bar;
use sonium_fem::physics::coupled::piezoelectric::PiezoMaterial;
use sonium_qa::{QaResult, QaStatus};
use std::f64::consts::PI;

const CASE_ID: &str = "fem.piezoelectric-1d-thickness-mode.v1";

#[derive(Debug)]
struct Metrics {
    frequency: f64,
    frequency_error: f64,
    field_error: f64,
    potential_error: f64,
    residual: f64,
    zero_mode: f64,
}

fn solve(n: usize, length: f64, material: &PiezoMaterial) -> Result<Metrics, String> {
    let rod = piezo_bar::assemble(n, length, material)?;
    let piezo_bar::RodModes {
        eigenvalues,
        displacements: modes,
        potentials,
    } = rod.open_circuit_modes()?;
    let c = material.stiffness[2][2];
    let e = material.piezo_stress[2][2];
    let epsilon = material.permittivity[2][2];
    let effective = c + e * e / epsilon;
    let mut order: Vec<usize> = (0..eigenvalues.len()).collect();
    order.sort_by(|&a, &b| eigenvalues[a].total_cmp(&eigenvalues[b]));
    let zero_mode = eigenvalues[order[0]].abs();
    let mode = order[1];
    let eigenvalue = eigenvalues[mode];
    if !eigenvalue.is_finite() || eigenvalue <= 0.0 || zero_mode > 1e-7 {
        return Err("invalid free-free piezoelectric spectrum".into());
    }
    let frequency = (eigenvalue * c / material.density).sqrt() / (2.0 * PI * length);
    // For free-free open electrodes D=0, cD=cE+e^2/epsilon. The first
    // extensional frequency is sqrt(cD/rho)/(2L), not the cE helper value.
    let exact_frequency = (effective / material.density).sqrt() / (2.0 * length);
    let mut state = vec![0.0; 2 * rod.nodes];
    for i in 0..rod.nodes {
        state[i] = modes[(i, mode)];
        state[rod.nodes + i] = potentials[mode][i];
    }
    let action = rod.action(&state);
    let mass_action: Vec<f64> = (0..rod.nodes)
        .map(|i| {
            (0..rod.nodes)
                .map(|j| rod.mass[i * rod.nodes + j] * state[j])
                .sum()
        })
        .collect();
    let numerator = (0..rod.nodes)
        .map(|i| (action[i] - eigenvalue * mass_action[i]).powi(2))
        .sum::<f64>()
        + (1..rod.nodes)
            .map(|i| action[rod.nodes + i].powi(2))
            .sum::<f64>();
    let denominator = action.iter().take(rod.nodes).map(|v| v * v).sum::<f64>()
        + mass_action
            .iter()
            .map(|v| (eigenvalue * v).powi(2))
            .sum::<f64>();
    let residual = (numerator / denominator).sqrt();
    let amplitude = (0..rod.nodes)
        .map(|i| state[i] * (PI * i as f64 / n as f64).cos())
        .sum::<f64>()
        / (0..rod.nodes)
            .map(|i| (PI * i as f64 / n as f64).cos().powi(2))
            .sum::<f64>();
    if !amplitude.is_finite() || amplitude.abs() < 1e-8 {
        return Err("piezoelectric mode has no longitudinal amplitude".into());
    }
    let coupling = e / (c * epsilon).sqrt();
    let potential_error = (0..rod.nodes)
        .map(|i| (state[rod.nodes + i] - coupling * (state[i] - state[0])).abs())
        .fold(0.0_f64, f64::max)
        / amplitude.abs();
    let mut error = 0.0;
    let mut norm = 0.0;
    for cell in 0..n {
        for xi in [
            (1.0 - 1.0 / 3.0_f64.sqrt()) / 2.0,
            (1.0 + 1.0 / 3.0_f64.sqrt()) / 2.0,
        ] {
            let actual = ((1.0 - xi) * state[cell] + xi * state[cell + 1]) / amplitude;
            let reference = (PI * (cell as f64 + xi) / n as f64).cos();
            error += (actual - reference).powi(2) / (2 * n) as f64;
            norm += reference.powi(2) / (2 * n) as f64;
        }
    }
    Ok(Metrics {
        frequency,
        frequency_error: (frequency / exact_frequency - 1.0).abs(),
        field_error: (error / norm).sqrt(),
        potential_error,
        residual,
        zero_mode,
    })
}

fn run() -> Result<QaResult, String> {
    let mut maximum_frequency_error = 0.0_f64;
    let mut maximum_field_error = 0.0_f64;
    let mut minimum_frequency_rate = f64::INFINITY;
    let mut minimum_field_rate = f64::INFINITY;
    let mut maximum_residual = 0.0_f64;
    let mut maximum_potential_error = 0.0_f64;
    let mut maximum_zero_mode = 0.0_f64;
    let mut stiffening_error = 0.0_f64;
    let mut minimum_frequency_increase = f64::INFINITY;
    for (material, length) in [
        (PiezoMaterial::pzt_5a(), 0.01),
        (PiezoMaterial::pzt_5h(), 0.005),
    ] {
        let mut metrics = Vec::new();
        for n in [8, 16, 32] {
            let m = solve(n, length, &material)?;
            println!("thickness n={n} L={length}: {m:?}");
            maximum_residual = maximum_residual.max(m.residual);
            maximum_potential_error = maximum_potential_error.max(m.potential_error);
            maximum_zero_mode = maximum_zero_mode.max(m.zero_mode);
            metrics.push(m);
        }
        if metrics.windows(2).any(|pair| {
            pair[0].frequency_error <= pair[1].frequency_error
                || pair[0].field_error <= pair[1].field_error
        }) {
            return Err("piezoelectric frequency/field errors do not contract".into());
        }
        let fine = &metrics[2];
        let medium = &metrics[1];
        maximum_frequency_error = maximum_frequency_error.max(fine.frequency_error);
        maximum_field_error = maximum_field_error.max(fine.field_error);
        minimum_frequency_rate =
            minimum_frequency_rate.min((medium.frequency_error / fine.frequency_error).log2());
        minimum_field_rate = minimum_field_rate.min((medium.field_error / fine.field_error).log2());
        let mut uncoupled = material.clone();
        uncoupled.piezo_stress[2][2] = 0.0;
        let control = solve(32, length, &uncoupled)?;
        maximum_residual = maximum_residual.max(control.residual);
        let c = material.stiffness[2][2];
        let e = material.piezo_stress[2][2];
        let epsilon = material.permittivity[2][2];
        let ratio = fine.frequency / control.frequency;
        let expected = (1.0 + e * e / (epsilon * c)).sqrt();
        stiffening_error = stiffening_error.max((ratio / expected - 1.0).abs());
        minimum_frequency_increase = minimum_frequency_increase.min(ratio - 1.0);
    }
    let mut record = QaResult::passed(CASE_ID)
        .requested_backend("cpu")
        .actual_backend("cpu");
    for (key, value, bound) in [
        ("frequency_error", maximum_frequency_error, 5e-4),
        ("modal_field_error", maximum_field_error, 0.002),
        ("coupled_eigen_residual", maximum_residual, 1e-9),
        (
            "open_circuit_potential_error",
            maximum_potential_error,
            1e-9,
        ),
        ("open_circuit_stiffening_error", stiffening_error, 1e-9),
        ("rigid_mode_eigenvalue", maximum_zero_mode, 1e-7),
    ] {
        if !value.is_finite() || value > bound {
            return Err(format!("{key} {value:.3e} exceeds {bound:.3e}"));
        }
        record = record.metric(key, value).maximum(key, bound);
    }
    if !minimum_frequency_rate.is_finite()
        || minimum_frequency_rate < 1.8
        || !minimum_field_rate.is_finite()
        || minimum_field_rate < 1.8
        || minimum_frequency_increase < 0.01
    {
        return Err("piezoelectric refinement or electrical stiffening gate failed".into());
    }
    Ok(record.metric("frequency_refinement_rate",minimum_frequency_rate).minimum("frequency_refinement_rate",1.8)
        .metric("modal_field_refinement_rate",minimum_field_rate).minimum("modal_field_refinement_rate",1.8)
        .metric("open_circuit_frequency_increase",minimum_frequency_increase).minimum("open_circuit_frequency_increase",0.01)
        .metric("solved_eigenproblems",8.0).minimum("solved_eigenproblems",8.0)
        .seed("pzt5a-pzt5h:free-free-open:n8-16-32:uncoupled-controls")
        .reference("free-free open circuit f=sqrt((cE+e^2/epsilon)/rho)/(2L), u=cos(pi x/L), phi=e/epsilon*(u-u(0)); uncoupled control")
        .reference_checksum("analytic:d33-open-circuit-thickness-mode:v1")
        .input_checksum("production-coupled-d33-and-consistent-mass:n8-16-32:pzt5a-pzt5h:v1")
        .dimension("1D longitudinal rod subspace, free mechanical ends, insulating electrical ends and one potential gauge")
        .formulation("production coupled piezoelectric blocks; numerical electrical Schur elimination; consistent-mass generalized eigensolve and full coupled residual")
        .tolerance("fine frequency <= 5e-4 and quadrature field <= 2e-3; second-order rates; electrical stiffening and coupled residual <= 1e-9"))
}

fn main() {
    let result = match run() {
        Ok(record) => record,
        Err(error) => QaResult::failed(CASE_ID, error),
    };
    match result.to_json_line() {
        Ok(json) => println!("QA_RESULT: {json}"),
        Err(error) => {
            eprintln!("FAILED: {error}");
            std::process::exit(1);
        }
    }
    if result.status != QaStatus::Passed {
        println!("FAILED");
        std::process::exit(1);
    }
    println!("SUCCESS");
}
