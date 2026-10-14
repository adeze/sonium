//! Measure the explicit full-stroke hollow rail without claiming native parity.
//!
//! Uses checked solid boundary surfaces and typed TYPE7 self-contact cards.
//! The original solid-column elastic reference is retained separately. Actual
//! full-duration folding,energy and native parity require subsequent checks.

// Rust guideline compliant 2026-02-21

#[path = "../bin/fem_opencourant_rail_crush.rs"]
#[expect(
    dead_code,
    reason = "Only the original fixture and its named plastic extension are used"
)]
mod original;

#[path = "../bin/oc_bench.rs"]
#[expect(
    dead_code,
    reason = "Uses only the complete native plastic rail history"
)]
mod native;

use sha2::{Digest, Sha256};
use sonium_qa::rad::{
    contact_surface::write_explicit_solid_contact_starter, engine_exe, run_extended_case,
    write_extended_engine,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let factor: usize = std::env::args().nth(1).map_or(Ok(1), |v| v.parse())?;
    let fixture = std::env::var("P2_RAIL_FIXTURE").unwrap_or_else(|_| "platens".into());
    let (case, surfaces, extra) = match fixture.as_str() {
        "platens" => original::rail_full_fixture::full_platen_case(factor)?,
        "grips" => original::rail_full_fixture::full_case(factor)?,
        _ => return Err("P2_RAIL_FIXTURE must be platens or grips".into()),
    };
    let work = std::env::temp_dir().join(format!("p2_rail_full_{factor}_{}", std::process::id()));
    std::fs::create_dir_all(&work)?;
    let starter = write_explicit_solid_contact_starter(&case, &surfaces, &extra)?;
    let engine = write_extended_engine(&case);
    if std::env::var("P2_RAIL_MODE").as_deref() == Ok("contact-initial") {
        return inspect_initial_contact(&case, &surfaces, &extra, &starter, &engine);
    }
    std::fs::write(work.join(format!("{}_0000.rad", case.base.name)), &starter)?;
    std::fs::write(work.join(format!("{}_0001.rad", case.base.name)), &engine)?;
    if std::env::var("P2_RAIL_MODE").as_deref() == Ok("native") {
        let timestep_scale: f64 = std::env::args().nth(2).map_or(Ok(1.0), |v| v.parse())?;
        let penalty_scale: f64 = std::env::args().nth(3).map_or(Ok(1.0), |v| v.parse())?;
        let profile_time = std::env::var("P2_RAIL_PROFILE_TIME_S")
            .ok()
            .map(|v| v.parse())
            .transpose()?;
        let history = native::rail_plastic_history(
            &work,
            &case,
            &surfaces,
            &extra,
            timestep_scale,
            penalty_scale,
            profile_time,
        )?;
        std::fs::write(
            std::env::var("P2_RAIL_OUTPUT")?,
            serde_json::to_vec_pretty(
                &serde_json::json!({"status":"complete full-stroke native rail diagnostic;QA parity and convergence unproven",
                "refinement_factor":factor,"fixture":fixture,"work_directory":work,
                "timestep_scale":timestep_scale,"penalty_scale":penalty_scale,"native":history,
                "scope":"Exact physical reference decks; objective Gauss B-bar LAW36 bulk, all-body unilateral frictionless platens and checked native full-boundary self-contact. No erosion or mass scaling. Normal contact timestep bound supplements bulk/wall limits; native/reference formulation and penalty/time/h convergence remain required."}),
            )?,
        )?;
        println!("Complete native full-stroke rail diagnostic;original QA gate unproven");
        return Ok(());
    }
    println!(
        "Full hollow rail:factor{factor},560*factor^3bricks,full1.2s/60%stroke;reference diagnostic starting"
    );
    let output = run_extended_case(&work, &engine_exe().ok_or("OpenCourant required")?, &case)?;
    if output.history.times.is_empty()
        || output
            .history
            .channels
            .values()
            .flatten()
            .any(|v| !v.is_finite())
    {
        return Err("complete finite full-stroke rail reference history required".into());
    }
    let result = serde_json::json!({"status":"complete full-stroke external diagnostic;folding/native parity/convergence unproven",
        "refinement_factor":factor,"fixture":fixture,"engine":output.engine,"work_directory":work,
        "physical_stop_time_s":case.base.stop_time_s,"stroke_m":0.12,
        "bricks":case.base.mesh.bricks.len(),"node_ids":case.base.th_nodes,
        "contact_boundary_quads":surfaces[0].quads.len(),
        "history":{"times_s":output.history.times,"channels":output.history.channels},
        "starter_sha256":format!("{:x}",Sha256::digest(starter.as_bytes())),
        "engine_deck_sha256":format!("{:x}",Sha256::digest(engine.as_bytes())),
        "scope":"Explicitnewplasticleg:originalnominalL.2m/outer.05m/E200GPa/nu.3/rho7800/speed.1m/s;2mmhollowwall,40umtrigger,Voce250MPa+300MPa*(1-exp(-5ep)),full1.2s/60%stroke,typedallboundaryTYPE7selfcontact. platens restoresoriginalfreebody/frictionlessanvil+crusherwithallstructuralnodeseligible;grips retainsoldclampedtip-onlycontrol. Actualplatencoverage/folding/energy/h-timeconvergence/nativeparity requireadmission. Originalelasticcolumnunchanged. Noerosion,massscaling,measured-alloy orQApassclaim."});
    std::fs::write(
        std::env::var("P2_RAIL_OUTPUT")?,
        serde_json::to_vec_pretty(&result)?,
    )?;
    println!("Complete full-stroke hollow rail reference;originalQA gate unproven");
    Ok(())
}

fn inspect_initial_contact(
    case: &sonium_qa::rad::RadExtendedCase,
    surfaces: &[sonium_qa::rad::contact_surface::RadSurfSegments],
    extra: &sonium_qa::rad::RadExtra,
    starter: &str,
    engine: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    use sonium_qa::rad::contact_surface::native_solid_self_contact;

    // Unit base stiffness is a geometry diagnostic. The declared Stfac is
    // applied by the checked adapter; it does not establish TYPE7 parity.
    let native = native_solid_self_contact(case, surfaces, extra, 1.0)?;
    let coordinates = native
        .coordinates
        .iter()
        .flatten()
        .copied()
        .collect::<Vec<_>>();
    let node_ids = native.node_ids;
    let mut contact = native.operator;
    let config = *contact.config();
    let mut forces = vec![0.0; coordinates.len()];
    let pairs = contact.apply_with_pairs(&coordinates, &vec![0.0; coordinates.len()], &mut forces);
    if forces.iter().any(|v| !v.is_finite()) || pairs.iter().any(|p| !p.gap.is_finite()) {
        return Err("initial native self-contact produced nonfinite values".into());
    }
    let maximum_penetration = pairs.iter().map(|p| -p.gap).fold(0.0, f64::max);
    let result = serde_json::json!({
        "status": "initial native self-contact diagnostic;full-stroke parity unproven",
        "bricks": case.base.mesh.bricks.len(), "triangles": 2 * surfaces[0].quads.len(),
        "slave_nodes": contact.num_slave_nodes(), "pair_count": pairs.len(),
        "maximum_penetration_m": maximum_penetration,
        "maximum_geometry_probe_force": forces.iter().map(|v| v.abs()).fold(0.0, f64::max),
        "force_sum": (0..3).map(|a| forces.iter().skip(a).step_by(3).sum::<f64>()).collect::<Vec<_>>(),
        "gap_offset_m": config.thickness_offset, "search_radius_m": config.search_radius,
        "penalty_n_per_m": config.penalty,
        "pairs": pairs.iter().map(|p| serde_json::json!({"node_id":node_ids[p.slave],"gap_m":p.gap,"normal":p.normal,"force":p.force})).collect::<Vec<_>>(),
        "starter_sha256": format!("{:x}", Sha256::digest(starter.as_bytes())),
        "engine_deck_sha256": format!("{:x}", Sha256::digest(engine.as_bytes())),
        "scope": "Actual production self-contact operator on undeformed full hollow rail with brick volume neighborhoods and reference material-edge paths bounded by search+gap. Quad faces split along their declared first-third diagonal. Unit base stiffness times declared Stfac is only a geometry diagnostic; TYPE7 stiffness, contact evolution, energy and full native parity remain unproven."
    });
    std::fs::write(
        std::env::var("P2_RAIL_OUTPUT")?,
        serde_json::to_vec_pretty(&result)?,
    )?;
    println!(
        "Initial native rail contact: {} pairs, maximum penetration {maximum_penetration:.12e}m",
        pairs.len()
    );
    if !pairs.is_empty() {
        return Err("undeformed full rail must have zero native self-contact pairs;diagnostic evidence retained".into());
    }
    Ok(())
}
