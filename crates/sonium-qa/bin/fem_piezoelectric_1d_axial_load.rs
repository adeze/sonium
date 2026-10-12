//! Solved coupled d33 rods with displacement, charge and reaction references.

#[expect(
    dead_code,
    reason = "Shared rod helper contains static and modal APIs; each binary exercises one."
)]
#[path = "support/piezo_bar.rs"]
mod piezo_bar;
use sonium_fem::physics::coupled::piezoelectric::PiezoMaterial;
use sonium_qa::{QaResult, QaStatus};

const CASE_ID: &str = "fem.piezoelectric-1d-axial-load.v1";

fn run() -> Result<QaResult, String> {
    let mut displacement_error = 0.0_f64;
    let mut potential_error = 0.0_f64;
    let mut stress_error = 0.0_f64;
    let mut charge_error = 0.0_f64;
    let mut reaction_error = 0.0_f64;
    let mut residual = 0.0_f64;
    let mut symmetry_error = 0.0_f64;
    let mut solved_cases = 0;
    let mut minimum_free_dofs = usize::MAX;
    for material in [PiezoMaterial::pzt_5a(), PiezoMaterial::pzt_5h()] {
        let c = material.stiffness[2][2];
        let e = material.piezo_stress[2][2];
        let epsilon = material.permittivity[2][2];
        for n in [8, 16, 32] {
            let rod = piezo_bar::assemble(n, 0.01, &material)?;
            let count = 2 * rod.nodes;
            symmetry_error = symmetry_error.max(rod.symmetry_error);
            // Loaded short-circuit, voltage-driven free, open-circuit loaded,
            // and voltage-driven blocked rods all retain interior unknowns.
            for (load_stress, voltage, open, blocked) in [
                (1e6, 100.0, false, false),
                (0.0, 100.0, false, false),
                (1e6, 0.0, true, false),
                (0.0, 100.0, false, true),
            ] {
                let effective = c + e * e / epsilon;
                let strain = if blocked {
                    0.0
                } else if open {
                    load_stress / effective
                } else {
                    (load_stress + e * voltage / rod.length) / c
                };
                let field = if open {
                    -e * strain / epsilon
                } else {
                    voltage / rod.length
                };
                let stress = if blocked { -e * field } else { load_stress };
                let displacement_flux = e * strain + epsilon * field;
                let strain_scale =
                    ((load_stress.abs() + e.abs() * field.abs()) / c).max(strain.abs());
                let field_scale = field.abs().max(e.abs() * strain_scale / epsilon);
                let flux_scale = e.abs() * strain_scale + epsilon * field_scale;
                let stress_scale = c * strain_scale + e.abs() * field_scale;
                let mut rhs = vec![0.0; count];
                if !blocked {
                    rhs[rod.nodes - 1] = load_stress / c;
                }
                let mut boundary = vec![(0, 0.0), (rod.nodes, 0.0)];
                if !open {
                    boundary.push((count - 1, -voltage / rod.phi_scale));
                }
                if blocked {
                    boundary.push((rod.nodes - 1, 0.0));
                }
                let free_dofs = count - boundary.len();
                if free_dofs == 0 {
                    return Err("coupled rod has no free unknowns".into());
                }
                minimum_free_dofs = minimum_free_dofs.min(free_dofs);
                let values = rod.solve(&rhs, &boundary)?;
                let action = rod.action(&values);
                for i in 0..rod.nodes {
                    let x = rod.length * i as f64 / n as f64;
                    displacement_error = displacement_error.max(
                        (values[i] * rod.length - strain * x).abs() / (strain_scale * rod.length),
                    );
                    potential_error = potential_error.max(
                        (values[rod.nodes + i] * rod.phi_scale + field * x).abs()
                            / (field_scale * rod.length),
                    );
                }
                for i in 0..n {
                    let actual_strain = (values[i + 1] - values[i]) * n as f64;
                    let actual_field = -(values[rod.nodes + i + 1] - values[rod.nodes + i])
                        * rod.phi_scale
                        * n as f64
                        / rod.length;
                    stress_error = stress_error
                        .max((c * actual_strain - e * actual_field - stress).abs() / stress_scale);
                    charge_error = charge_error.max(
                        (e * actual_strain + epsilon * actual_field - displacement_flux).abs()
                            / flux_scale,
                    );
                }
                // The electrical reaction is outward D flux at the electrode,
                // with the opposite sign at the other end; no helper formula
                // or imposed charge replaces this unmodified block action.
                for (i, sign) in [(rod.nodes, -1.0), (count - 1, 1.0)] {
                    let flux = action[i] * (c * epsilon).sqrt();
                    charge_error =
                        charge_error.max((flux - sign * displacement_flux).abs() / flux_scale);
                }
                for (i, sign) in [(0, -1.0), (rod.nodes - 1, 1.0)] {
                    reaction_error =
                        reaction_error.max((action[i] * c - sign * stress).abs() / stress_scale);
                }
                let mut squared = 0.0;
                for i in 0..count {
                    if !boundary.iter().any(|&(j, _)| i == j) {
                        squared += (action[i] - rhs[i]).powi(2);
                    }
                }
                let scale = action.iter().map(|v| v * v).sum::<f64>().sqrt();
                if !scale.is_finite() || scale <= 0.0 {
                    return Err("coupled rod carries no physical load/reaction".into());
                }
                residual = residual.max(squared.sqrt() / scale);
                for &(i, value) in &boundary {
                    if (values[i] - value).abs() > 1e-12 {
                        return Err("coupled rod BC violated".into());
                    }
                }
                solved_cases += 1;
            }
        }
    }
    let mut record = QaResult::passed(CASE_ID)
        .requested_backend("cpu")
        .actual_backend("cpu");
    for (key, value) in [
        ("displacement_error", displacement_error),
        ("potential_error", potential_error),
        ("stress_error", stress_error),
        ("charge_flux_error", charge_error),
        ("mechanical_reaction_error", reaction_error),
        ("physical_residual", residual),
        ("coupled_symmetry_error", symmetry_error),
    ] {
        if !value.is_finite() || value > 1e-9 {
            return Err(format!("{key} {value:.3e} exceeds 1e-9"));
        }
        record = record.metric(key, value).maximum(key, 1e-9);
    }
    Ok(record.metric("solved_cases",solved_cases as f64).minimum("solved_cases",24.0)
        .seed("d33:pzt5a-pzt5h:n8-16-32:short-loaded-free-open-blocked")
        .reference("independent T=cS-eE, D=eS+epsilon E; open D=0 gives cD=cE+e^2/epsilon; exact uniform rod fields")
        .reference_checksum("analytic:d33-rod-stress-charge:v1")
        .input_checksum("production-d33-blocks:rod0.01m:two-pzt-materials:24-solves:v1")
        .dimension("1D longitudinal rod subspace of triangular strip, interior displacement and potential unknowns")
        .formulation("production piezoelectric stiffness/coupling/dielectric assembly, coupled sparse solve, open/short electrical and free/blocked mechanical BCs")
        .tolerance("affine coupled rod displacement/potential/stress/charge and reaction errors <= 1e-9"))
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
