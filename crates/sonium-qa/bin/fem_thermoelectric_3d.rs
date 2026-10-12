//! Form contract and solved temperature-linearized 3D thermoelectric MMS.

use sonium_fem::assembly::{TripletMatrix, assemble_linear};
use sonium_fem::basis::PolynomialDegree;
use sonium_fem::mesh::{Mesh, unit_cube_tetrahedra};
use sonium_fem::physics::thermal::heat_conduction::HeatSourceForm;
use sonium_fem::physics::thermal::thermoelectric::{
    ElectricalConductionForm, JouleHeatSourceForm, ThermoelectricMaterial,
    assemble_thermoelectric_system,
};
use sonium_fem::quadrature::QuadratureRule;
use sonium_fem::spaces::{FunctionSpace, H1Space};
use sonium_qa::{QaResult, QaStatus};
use sonium_solvers::SparseLuFactorization;
use std::collections::HashMap;

const CASE_ID: &str = "fem.thermoelectric-3d-mms.v1";
const ORDER: usize = 4;
const LINEARIZATION_TEMPERATURE: f64 = 300.0;
const MATERIAL: ThermoelectricMaterial = ThermoelectricMaterial {
    electrical_conductivity: 2.0,
    thermal_conductivity: 1.5,
    seebeck_coefficient: 0.005,
    seebeck_temperature_coefficient: 0.0,
    reference_temperature: 300.0,
    density: 1.0,
    specific_heat: 1.0,
};

fn reference(x: f64, y: f64, z: f64) -> (f64, [f64; 3], f64, [f64; 3]) {
    (
        1.0 + 0.1
            * (x * (1.0 - x)
                + 2.0 * y * (1.0 - y)
                + 3.0 * z * (1.0 - z)
                + x * y
                + 2.0 * x * z
                + y * z),
        [
            0.1 * (1.0 - 2.0 * x + y + 2.0 * z),
            0.1 * (2.0 - 4.0 * y + x + z),
            0.1 * (3.0 - 6.0 * z + 2.0 * x + y),
        ],
        10.0 * (x * (1.0 - x) + y * (1.0 - y) + 2.0 * z * (1.0 - z) + x * y + x * z + y * z),
        [
            10.0 * (1.0 - 2.0 * x + y + z),
            10.0 * (1.0 - 2.0 * y + x + z),
            10.0 * (2.0 - 4.0 * z + x + y),
        ],
    )
}
// Independently differentiate the declared polynomial fields. Lap(V)=-1.2
// and Lap(theta)=-80. Alpha=.005, Pi=alpha*300=1.5, sigma=2, k=1.5.
fn current_source(_: f64, _: f64, _: f64) -> f64 {
    2.0 * 1.2 + 0.01 * 80.0
}
fn heat_source(x: f64, y: f64, z: f64) -> f64 {
    let (_, gradient, _, _) = reference(x, y, z);
    3.0 * 1.2 + 1.515 * 80.0 - 2.0 * gradient.iter().map(|g| g * g).sum::<f64>()
}

fn joule_load(mesh: &Mesh, space: &H1Space, values: &[f64]) -> Vec<f64> {
    // Production source callbacks receive actual gradients reconstructed from
    // the solved FE voltage, at the same quadrature points used by assembly.
    let mut gradients = HashMap::new();
    for element in 0..mesh.num_elements() {
        let dofs = space.local_to_global(mesh, element);
        for qp in QuadratureRule::new(mesh.elements[element].element_type, ORDER).iter() {
            let tab = space.tabulate(mesh, element, qp.xi(), qp.eta(), qp.zeta());
            let gradient: [f64; 3] = std::array::from_fn(|axis| {
                dofs.iter()
                    .enumerate()
                    .map(|(i, &d)| values[d] * tab.grad(i, axis))
                    .sum()
            });
            gradients.insert(tab.physical_coords.map(f64::to_bits), gradient);
        }
    }
    assemble_linear(
        &JouleHeatSourceForm {
            electric_field: |x: f64, y: f64, z: f64| {
                gradients[&[x.to_bits(), y.to_bits(), z.to_bits()]].map(|g| -g)
            },
            electrical_conductivity: |_, _, _| MATERIAL.electrical_conductivity,
        },
        space,
        mesh,
        ORDER,
    )
}

#[derive(Debug, Clone, Copy)]
enum Control {
    Full,
    NoSeebeck,
    NoPeltier,
    NoJoule,
}
#[derive(Debug)]
struct Metrics {
    voltage: f64,
    temperature: f64,
    voltage_gradient: f64,
    temperature_gradient: f64,
    residual: f64,
    boundary: f64,
    kelvin_error: f64,
    iterations: usize,
    free_dofs: usize,
}

fn solve(n: usize, degree: PolynomialDegree, control: Control) -> Result<Metrics, String> {
    let mut mesh = unit_cube_tetrahedra(n);
    for face in &mut mesh.boundaries {
        face.marker = 1;
    }
    let space = H1Space::new(degree);
    let count = space.num_global_dofs(&mesh);
    let coordinates = space
        .dof_coordinates(&mesh)
        .ok_or("thermoelectric H1 coordinates missing")?;
    let mut blocks =
        assemble_thermoelectric_system(&mesh, degree, MATERIAL, LINEARIZATION_TEMPERATURE);
    let kelvin_scale = blocks
        .block_10
        .values
        .iter()
        .map(|v| v.abs())
        .fold(0.0_f64, f64::max);
    if kelvin_scale == 0.0 {
        return Err("thermoelectric coupling blocks are empty".into());
    }
    let kelvin_error = blocks
        .block_10
        .values
        .iter()
        .zip(&blocks.block_01.values)
        .map(|(p, s)| (p - LINEARIZATION_TEMPERATURE * s).abs() / kelvin_scale)
        .fold(0.0_f64, f64::max);
    match control {
        Control::NoSeebeck => blocks.block_01.values.fill(0.0),
        Control::NoPeltier => {
            blocks.block_10.values.fill(0.0);
            // Remove the induced Pi*sigma*alpha contribution as well.
            for v in &mut blocks.block_11.values {
                *v *= 1.5 / 1.515;
            }
        }
        Control::Full | Control::NoJoule => {}
    }
    let original = blocks.to_monolithic();
    let raw = original.to_csr();
    let mut constrained = vec![false; 2 * count];
    let mut trace = vec![0.0; 2 * count];
    for i in space.boundary_dofs(&mesh, 1) {
        let p = coordinates[i];
        let (v, _, theta, _) = reference(p[0], p[1], p[2]);
        constrained[i] = true;
        constrained[count + i] = true;
        trace[i] = v;
        trace[count + i] = theta;
    }
    let free_dofs = constrained.iter().filter(|&&c| !c).count();
    if free_dofs == 0 {
        return Err("thermoelectric system has no free unknowns".into());
    }
    // Apply exact column lifting to both fields in the monolithic system.
    let mut admitted = TripletMatrix::new(2 * count, 2 * count);
    let mut lifting = vec![0.0; 2 * count];
    for ((&row, &col), &value) in original
        .rows
        .iter()
        .zip(&original.cols)
        .zip(&original.values)
    {
        if constrained[row] {
            continue;
        }
        if constrained[col] {
            lifting[row] -= value * trace[col];
        } else {
            admitted.push(row, col, value);
        }
    }
    for (i, &c) in constrained.iter().enumerate() {
        if c {
            admitted.push(i, i, 1.0);
        }
    }
    let factor = SparseLuFactorization::factorize(&admitted.to_csr()).map_err(|e| e.to_string())?;
    let current = assemble_linear(
        &HeatSourceForm {
            source_fn: current_source,
        },
        &space,
        &mesh,
        ORDER,
    );
    let external = assemble_linear(
        &HeatSourceForm {
            source_fn: heat_source,
        },
        &space,
        &mesh,
        ORDER,
    );
    let physical_rhs = |heat: &[f64]| {
        let mut rhs = current.clone();
        rhs.extend(external.iter().zip(heat).map(|(a, b)| a + b));
        rhs
    };
    let mut values = vec![0.0; 2 * count];
    let mut iterations = 0;
    let mut converged = false;
    for iteration in 1..=40 {
        let heat = if matches!(control, Control::NoJoule) {
            vec![0.0; count]
        } else {
            joule_load(&mesh, &space, &values[..count])
        };
        let rhs = physical_rhs(&heat);
        let rhs: Vec<f64> = rhs
            .iter()
            .enumerate()
            .map(|(i, &r)| {
                if constrained[i] {
                    trace[i]
                } else {
                    r + lifting[i]
                }
            })
            .collect();
        let next = factor.solve(&rhs).map_err(|e| e.to_string())?;
        if next.iter().any(|v| !v.is_finite()) {
            return Err("nonfinite thermoelectric state".into());
        }
        // Admit convergence using each field's own scale, rather than allowing
        // the much larger temperature-rise norm to hide voltage changes.
        let mut change = 0.0_f64;
        for offset in [0, count] {
            let numerator = (offset..offset + count)
                .map(|i| (next[i] - values[i]).powi(2))
                .sum::<f64>();
            let denominator = (offset..offset + count)
                .map(|i| next[i] * next[i])
                .sum::<f64>();
            change = change.max((numerator / denominator).sqrt());
        }
        values = next;
        iterations = iteration;
        if change < 1e-12 {
            converged = true;
            break;
        }
    }
    if !converged {
        return Err("thermoelectric recovered-field Joule iteration did not converge".into());
    }
    let heat = if matches!(control, Control::NoJoule) {
        vec![0.0; count]
    } else {
        joule_load(&mesh, &space, &values[..count])
    };
    let rhs = physical_rhs(&heat);
    let action = raw.matvec(&values);
    let mut residual = 0.0_f64;
    for offset in [0, count] {
        let numerator = (offset..offset + count)
            .filter(|&i| !constrained[i])
            .map(|i| (action[i] - rhs[i]).powi(2))
            .sum::<f64>();
        let denominator = (offset..offset + count)
            .filter(|&i| !constrained[i])
            .map(|i| rhs[i].powi(2))
            .sum::<f64>();
        residual = residual.max((numerator / denominator).sqrt());
    }
    let boundary = (0..2 * count)
        .filter(|&i| constrained[i])
        .map(|i| (values[i] - trace[i]).abs())
        .fold(0.0_f64, f64::max);
    let mut errors = [0.0; 4];
    let mut norms = [0.0; 4];
    for element in 0..mesh.num_elements() {
        let dofs = space.local_to_global(&mesh, element);
        for qp in QuadratureRule::new(mesh.elements[element].element_type, 5).iter() {
            let tab = space.tabulate(&mesh, element, qp.xi(), qp.eta(), qp.zeta());
            let p = tab.physical_coords;
            let (v, dv, theta, dt) = reference(p[0], p[1], p[2]);
            let weight = qp.weight * tab.det_j;
            let actual_v: f64 = dofs
                .iter()
                .enumerate()
                .map(|(i, &d)| values[d] * tab.value(i))
                .sum();
            let actual_t: f64 = dofs
                .iter()
                .enumerate()
                .map(|(i, &d)| values[count + d] * tab.value(i))
                .sum();
            errors[0] += weight * (actual_v - v).powi(2);
            norms[0] += weight * v * v;
            errors[1] += weight * (actual_t - theta).powi(2);
            norms[1] += weight * theta * theta;
            for axis in 0..3 {
                let gv: f64 = dofs
                    .iter()
                    .enumerate()
                    .map(|(i, &d)| values[d] * tab.grad(i, axis))
                    .sum();
                let gt: f64 = dofs
                    .iter()
                    .enumerate()
                    .map(|(i, &d)| values[count + d] * tab.grad(i, axis))
                    .sum();
                errors[2] += weight * (gv - dv[axis]).powi(2);
                norms[2] += weight * dv[axis].powi(2);
                errors[3] += weight * (gt - dt[axis]).powi(2);
                norms[3] += weight * dt[axis].powi(2);
            }
        }
    }
    Ok(Metrics {
        voltage: (errors[0] / norms[0]).sqrt(),
        temperature: (errors[1] / norms[1]).sqrt(),
        voltage_gradient: (errors[2] / norms[2]).sqrt(),
        temperature_gradient: (errors[3] / norms[3]).sqrt(),
        residual,
        boundary,
        kelvin_error,
        iterations,
        free_dofs,
    })
}

fn run() -> Result<QaResult, String> {
    let mut levels = Vec::new();
    for n in [4, 8, 16] {
        let m = solve(n, PolynomialDegree::P1, Control::Full)?;
        println!("thermoelectric P1 n={n}: {m:?}");
        levels.push(m);
    }
    let patch = solve(2, PolynomialDegree::P2, Control::Full)?;
    println!("thermoelectric P2 exact patch: {patch:?}");
    let seebeck = solve(2, PolynomialDegree::P2, Control::NoSeebeck)?;
    let peltier = solve(2, PolynomialDegree::P2, Control::NoPeltier)?;
    let joule = solve(2, PolynomialDegree::P2, Control::NoJoule)?;
    println!("thermoelectric no Seebeck: {seebeck:?}");
    println!("thermoelectric no Peltier: {peltier:?}");
    println!("thermoelectric no Joule: {joule:?}");
    if !levels.windows(2).all(|p| {
        p[0].voltage > p[1].voltage
            && p[0].temperature > p[1].temperature
            && p[0].voltage_gradient > p[1].voltage_gradient
            && p[0].temperature_gradient > p[1].temperature_gradient
    }) {
        return Err("thermoelectric fields and gradients do not contract".into());
    }
    let mut r = QaResult::passed(CASE_ID)
        .requested_backend("cpu")
        .actual_backend("cpu");
    for (name, value, bound) in [
        ("fine_voltage_error", levels[2].voltage, 0.005),
        ("fine_temperature_error", levels[2].temperature, 0.02),
        (
            "fine_voltage_gradient_error",
            levels[2].voltage_gradient,
            0.1,
        ),
        (
            "fine_temperature_gradient_error",
            levels[2].temperature_gradient,
            0.15,
        ),
        (
            "quadratic_patch_error",
            patch
                .voltage
                .max(patch.temperature)
                .max(patch.voltage_gradient)
                .max(patch.temperature_gradient),
            1e-9,
        ),
        (
            "physical_residual",
            levels
                .iter()
                .chain([&patch])
                .map(|m| m.residual)
                .fold(0.0_f64, f64::max),
            1e-9,
        ),
        (
            "boundary_error",
            levels
                .iter()
                .chain([&patch])
                .map(|m| m.boundary)
                .fold(0.0_f64, f64::max),
            1e-10,
        ),
        (
            "kelvin_block_error",
            levels
                .iter()
                .chain([&patch])
                .map(|m| m.kelvin_error)
                .fold(0.0_f64, f64::max),
            1e-12,
        ),
    ] {
        if !value.is_finite() || value > bound {
            return Err(format!("{name}: {value:.3e}>{bound:.3e}"));
        }
        r = r.metric(name, value).maximum(name, bound);
    }
    for (name, value, bound) in [
        (
            "voltage_rate",
            (levels[1].voltage / levels[2].voltage).log2(),
            1.8,
        ),
        (
            "temperature_rate",
            (levels[1].temperature / levels[2].temperature).log2(),
            1.8,
        ),
        (
            "voltage_gradient_rate",
            (levels[1].voltage_gradient / levels[2].voltage_gradient).log2(),
            0.8,
        ),
        (
            "temperature_gradient_rate",
            (levels[1].temperature_gradient / levels[2].temperature_gradient).log2(),
            0.8,
        ),
        ("omitted_seebeck_voltage_error", seebeck.voltage, 0.001),
        (
            "omitted_peltier_temperature_error",
            peltier.temperature,
            0.001,
        ),
        ("omitted_joule_temperature_error", joule.temperature, 1e-5),
    ] {
        if !value.is_finite() || value < bound {
            return Err(format!("{name}: {value:.3e}<{bound:.3e}"));
        }
        r = r.metric(name, value).minimum(name, bound);
    }
    Ok(r.metric("minimum_free_dofs",levels.iter().chain([&patch]).map(|m|m.free_dofs).min().unwrap_or(0) as f64)
        .minimum("minimum_free_dofs",1.0)
        .metric("minimum_source_iterations",levels.iter().chain([&patch]).map(|m|m.iterations).min().unwrap_or(0) as f64)
        .minimum("minimum_source_iterations",3.0)
        .seed("3d-mixed-xyz:alpha0.005:Pi1.5:sigma2:k1.5:P1n4-8-16:P2n2:3-controls")
        .input_checksum("production-monolithic-seebeck-peltier:solved-field-joule:v1")
        .reference("independent polynomial strong equations: sigma(-Lap V-alpha Lap theta)=fI; -Pi sigma Lap V-(k+Pi sigma alpha)Lap theta=fH+sigma|grad V|^2")
        .reference_checksum("analytic:3d-linearized-thermoelectric-joule-mms:v1")
        .dimension("3D unit cube; mixed xyz voltage and temperature rise; two unconstrained interior fields")
        .formulation("production temperature-linearized Seebeck/Peltier monolithic operator at 300K, recovered-field Joule source, coupled sparse LU, P1 refinement and P2 reproduction")
        .tolerance("field rates>=1.8, gradient rates>=0.8; P2 fields/gradients<=1e-9; original nonlinear-source residual<=1e-9; three missing-term controls"))
}

fn main() {
    sonium_qa::fem_physics_3d::run_bilinear_case(
        "fem.thermoelectric-3d.v1",
        &ElectricalConductionForm {
            electrical_conductivity: 1.0,
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
