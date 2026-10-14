//! Shared complete copper Taylor fixture and reference rendering.

// Rust guideline compliant 2026-02-21

use sonium_qa::rad::{
    RadExtendedCase, RadJohnsonCook, RadMaterial, RadPlasticMaterial, RadProbe, RadThRequest,
    bric_channel, probes_json, write_extended_engine, write_extended_starter,
};
use std::path::Path;

pub(crate) const NAME: &str = "taylor_bar_full_plastic";
// About six original elastic release-wave round trips; 4 us misses even one.
pub(crate) const STOP: f64 = 80e-6;

/// Build the complete isothermal copper impact with original physical inputs.
///
/// # Panics
/// Panics if the refinement factor is outside 1..=4.
pub(crate) fn full_case(factor: usize) -> RadExtendedCase {
    assert!(
        (1..=4).contains(&factor),
        "Taylor mesh factor must be 1..=4"
    );
    let mut base = super::taylor_bar_case();
    if factor > 1 {
        let (mut mesh, face, center) = super::cylinder_mesh(8 * factor, 6 * factor);
        let bar_nodes = (1..=mesh.num_nodes()).collect();
        let driver = mesh.add_node(base.walls[0].point_m);
        base.mesh = mesh;
        base.node_sets[0].nodes = face;
        base.node_sets[1].nodes = bar_nodes;
        base.th_nodes = vec![center];
        base.walls[0].driver_node = Some(driver);
    }
    base.name = NAME.into();
    base.title = "original copper Taylor bar: full isothermal plastic impact".into();
    base.stop_time_s = STOP;
    base.th_period_s = 1e-7;
    base.anim_period_s = 5e-6;
    base.functions[0].points = vec![(0.0, 1.0), (STOP, 1.0)];
    // The original final node drives the wall and carries no bar mass.
    let bar_nodes = base.node_sets[1].nodes.clone();
    let face_center = base.th_nodes[0];
    let face_edge = *base.node_sets[0]
        .nodes
        .iter()
        .max_by(|&&a, &&b| {
            let radius = |id: usize| {
                let p = base.mesh.nodes[id - 1];
                p[0].hypot(p[1])
            };
            radius(a).total_cmp(&radius(b))
        })
        .expect("original impact face");
    let tail = *bar_nodes
        .iter()
        .max_by(|&&a, &&b| base.mesh.nodes[a - 1][2].total_cmp(&base.mesh.nodes[b - 1][2]))
        .expect("original bar nodes");
    // Expected values are unused diagnostic placeholders, never QA admission.
    base.probes = [
        format!("DZ/{face_center}"),
        format!("DX/{face_edge}"),
        format!("DZ/{tail}"),
        bric_channel("PLAS", 1),
    ]
    .into_iter()
    .map(|channel| RadProbe {
        label: channel.clone(),
        channel,
        expected: 0.0,
        abs_tol: 0.0,
        rel_tol: 0.0,
    })
    .collect();
    base.th_nodes = bar_nodes;
    let mut case = RadExtendedCase::wrap_elastic(base);
    let elastic = case.base.material;
    case.material = RadMaterial::Plastic(RadPlasticMaterial::JohnsonCook(RadJohnsonCook {
        youngs_modulus_pa: elastic.youngs_modulus_pa,
        poisson_ratio: elastic.poisson_ratio,
        density_kg_per_m3: elastic.density_kg_per_m3,
        yield_pa: 90e6,
        hardening_pa: 292e6,
        hardening_exp: 0.31,
        rate_coeff: 0.025,
        ref_strain_rate: 1.0,
        // No erosion model is part of this intact copper mushrooming test.
        fail_plastic_strain: 1e20,
        max_stress_pa: 1e20,
    }));
    case.th_brick = RadThRequest::brick_stress((1..=case.base.mesh.bricks.len()).collect());
    case
}

/// Write complete physical decks with an explicit reference formulation.
///
/// # Errors
/// Rejects invalid rendering or failed directory/deck writes.
pub(crate) fn write_decks(
    case: &RadExtendedCase,
    work: &Path,
    viscous: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    std::fs::create_dir_all(work)?;
    let starter = isothermal_starter(case)?;
    let starter = if viscous {
        viscous_starter(&starter)?
    } else {
        starter
    };
    std::fs::write(work.join(format!("{NAME}_0000.rad")), starter)?;
    std::fs::write(
        work.join(format!("{NAME}_0001.rad")),
        write_extended_engine(case),
    )?;
    std::fs::write(
        work.join("probes.json"),
        probes_json(&case.base, "diagnostic only"),
    )?;
    Ok(())
}

/// Render isothermal copper while retaining all mechanical input fields.
///
/// # Errors
/// Rejects missing or unexpected Johnson-Cook thermal fields.
pub(crate) fn isothermal_starter(
    case: &RadExtendedCase,
) -> Result<String, Box<dyn std::error::Error>> {
    let mut lines = write_extended_starter(case)
        .lines()
        .map(str::to_owned)
        .collect::<Vec<_>>();
    let material = lines
        .iter()
        .position(|s| s == "/MAT/LAW2/1/1")
        .ok_or("full Taylor starter requires the original LAW2 material card")?;
    let thermal = material + 6;
    let mut fields = lines
        .get(thermal)
        .ok_or("LAW2 thermal row missing")?
        .split_whitespace()
        .map(str::to_owned)
        .collect::<Vec<_>>();
    if fields.len() != 5 {
        return Err("unexpected LAW2 thermal row layout".into());
    }
    // LAW2 is adiabatic without /HEAT when rho*Cp is nonzero. Explicit zero
    // heat capacity fixes T=Tr (Altair LAW2 comment 11), matching the native
    // isothermal parser. Retain every mechanical and rate-dependent field.
    fields[2] = "0.000000000000E0".into();
    lines[thermal] = fields
        .iter()
        .map(|s| format!("{s:>20}"))
        .collect::<String>();
    Ok(lines.join("\n") + "\n")
}

/// Select the standard viscous reference without changing physical inputs.
///
/// This is a controlled formulation diagnostic. Native and external hourglass
/// coefficients are not claimed equal: native uses rho*c*L_min^2 scaling,
/// whereas the reference uses its documented volume and linear/quadratic terms.
///
/// # Errors
/// Rejects a missing or changed original HEPH property row.
pub(crate) fn viscous_starter(starter: &str) -> Result<String, Box<dyn std::error::Error>> {
    let mut lines = starter.lines().map(str::to_owned).collect::<Vec<_>>();
    let row = lines
        .iter()
        .position(|s| s == "/PROP/SOLID/1/1")
        .ok_or("original solid property missing")?
        + 2;
    if lines
        .get(row)
        .ok_or("original solid property row missing")?
        .split_whitespace()
        .collect::<Vec<_>>()
        != ["24", "2"]
    {
        return Err("original HEPH formulation changed".into());
    }
    // PROP/TYPE14 Isolid1 is one-point Belytschko viscous stabilization.
    // Preserve Ismstr2 and every geometry, material, wall and output field.
    lines[row] = format!("{:>10}{:>10}", 1, 2);
    Ok(lines.join("\n") + "\n")
}
