//! Measure original full Hertz-fixture contact histories without asserting parity.
//!
//! The original 10 mm mesh does not resolve the continuum Hertz patch. This
//! diagnostic compares the actual three-dimensional discrete contact fixture;
//! it does not claim continuum Hertz force accuracy.

// Rust guideline compliant 2026-02-21

#[path = "../bin/support/hertz_native.rs"]
#[expect(
    dead_code,
    reason = "Only the actual native elastic/contact path is used"
)]
mod native;

use native::full_gauss_starter;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let destination = std::env::var("P2_HERTZ_OUTPUT")?;
    if std::env::var("P2_HERTZ_MODE").as_deref() == Ok("external") {
        use sha2::{Digest, Sha256};
        use sonium_qa::rad::{
            engine_exe, hertz_sphere_case, run_case, write_engine, write_starter,
        };
        let case = hertz_sphere_case();
        let mut starter = write_starter(&case);
        let engine = write_engine(&case);
        let decks = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("decks/hertz_sphere");
        if std::fs::read_to_string(decks.join("hertz_sphere_0000.rad"))? != starter
            || std::fs::read_to_string(decks.join("hertz_sphere_0001.rad"))? != engine
        {
            return Err("original Hertz decks differ from physical fixture writer".into());
        }
        let full_gauss = std::env::var("P2_HERTZ_FULL_GAUSS").as_deref() == Ok("1");
        if full_gauss {
            starter = full_gauss_starter(&starter)?;
        }
        let work = std::path::PathBuf::from(std::env::var("P2_HERTZ_WORK")?);
        std::fs::create_dir_all(&work)?;
        std::fs::write(work.join("hertz_sphere_0000.rad"), &starter)?;
        std::fs::write(work.join("hertz_sphere_0001.rad"), &engine)?;
        let exe = engine_exe().ok_or("OpenCourant required for original contact reference")?;
        let measured = run_case(&work, &exe, &case)?;
        let channels = vec![
            case.probes[0].channel.clone(),
            "FNZ/RWALL/1".into(),
            "FNZ/RWALL/2".into(),
        ];
        let values = channels
            .iter()
            .map(|c| measured.history.channel(c).map(|v| v.to_vec()))
            .collect::<Result<Vec<_>, _>>()?;
        let result = serde_json::json!({"status":"original full external contact measurement; QA parity unproven",
            "solid_formulation":if full_gauss {17} else {24},
            "engine":measured.engine,"times_s":measured.history.times,"probe_channels":channels,"probe_values":values,
            "starter_sha256":format!("{:x}",Sha256::digest(starter.as_bytes())),
            "engine_deck_sha256":format!("{:x}",Sha256::digest(engine.as_bytes()))});
        std::fs::write(destination, serde_json::to_vec_pretty(&result)?)?;
    } else {
        let scale = std::env::args().nth(1).map_or(Ok(1.0), |v| v.parse())?;
        let measured = native::original_hertz_history(scale)?;
        std::fs::write(destination, serde_json::to_vec_pretty(&measured)?)?;
    }
    println!("Original full discrete sphere contact measured; diagnostic, QA parity unproven");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn invalid_cfl_scale_is_rejected_before_execution() {
        for scale in [0.0, -1.0, 2.0, f64::NAN, f64::INFINITY] {
            assert!(native::original_hertz_history(scale).is_err());
        }
    }

    #[test]
    fn full_gauss_reference_changes_only_quadrature_property() {
        let case = sonium_qa::rad::hertz_sphere_case();
        let original = sonium_qa::rad::write_starter(&case);
        let gauss = full_gauss_starter(&original).unwrap();
        let original = original.lines().collect::<Vec<_>>();
        let gauss = gauss.lines().collect::<Vec<_>>();
        assert_eq!(original.len(), gauss.len());
        let changed = original
            .iter()
            .zip(&gauss)
            .enumerate()
            .filter(|(_, (a, b))| a != b)
            .collect::<Vec<_>>();
        assert_eq!(changed.len(), 1);
        let row = changed[0].0;
        assert_eq!(original[row - 2], "/PROP/SOLID/1/1");
        assert_eq!(
            gauss[row].split_whitespace().collect::<Vec<_>>(),
            ["17", "2", "0", "3"]
        );
    }
}
