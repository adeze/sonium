//! Compare original LAW1 rail observations without admitting full crush parity.

// Rust guideline compliant 2026-02-21

#[path = "../bin/oc_bench.rs"]
#[expect(dead_code, reason = "Uses the native elastic rail observation API")]
mod native;
#[path = "../bin/fem_opencourant_rail_crush.rs"]
#[expect(dead_code, reason = "Uses the frozen original rail fixture")]
mod original;

use sonium_qa::rad::{engine_exe, probes_json, run_case, write_engine, write_starter};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let scale: f64 = std::env::args().nth(1).map_or(Ok(1.0), |v| v.parse())?;
    // The original high elastic load needs a stiff wall. Refinements must
    // demonstrate penalty convergence, rather than accepting a force match.
    let penalty: f64 = std::env::args().nth(2).map_or(Ok(100.0), |v| v.parse())?;
    let case = original::rail_crush_case();
    let work = std::env::temp_dir().join(format!("p2_rail_elastic_{}", std::process::id()));
    std::fs::create_dir_all(&work)?;
    std::fs::write(work.join("rail_crush_0000.rad"), write_starter(&case))?;
    std::fs::write(work.join("rail_crush_0001.rad"), write_engine(&case))?;
    std::fs::write(
        work.join("probes.json"),
        probes_json(&case, "diagnostic only"),
    )?;
    let measured = if std::env::var("P2_RAIL_ELASTIC_MODE").as_deref() == Ok("reference") {
        let output = run_case(&work, &engine_exe().ok_or("OpenCourant required")?, &case)?;
        if output.history.times.is_empty()
            || output
                .history
                .times
                .last()
                .is_none_or(|t| *t < case.stop_time_s - 1.5 * case.th_period_s)
            || case.probes.iter().any(|probe| {
                output.history.channels.get(&probe.channel).is_none_or(|v| {
                    v.len() != output.history.times.len() || v.iter().any(|x| !x.is_finite())
                })
            })
        {
            return Err("complete finite original elastic reference observations required".into());
        }
        serde_json::json!({"status":"complete original elastic reference;fullcrush/nativeparity unproven",
            "physical_stop_time_s":case.stop_time_s,"work_directory":work,
            "history":{"times_s":output.history.times,"channels":output.history.channels}})
    } else {
        serde_json::json!({"status":"complete original native elastic diagnostic;fullcrush/parity unproven",
            "physical_stop_time_s":case.stop_time_s,"work_directory":work,
            "penalty_scale":penalty,"timestep_scale":scale,
            "native":native::rail_elastic_history(&work,scale,penalty)?})
    };
    std::fs::write(
        std::env::var("P2_RAIL_ELASTIC_OUTPUT")?,
        serde_json::to_vec_pretty(&measured)?,
    )?;
    Ok(())
}
