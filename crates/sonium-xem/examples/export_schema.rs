//! Export the canonical XEM schemas for release artifacts and external tools.
//!
//! The schemas are generated from the same Rust functions used by validation,
//! ensuring a published artifact cannot silently drift from the reader.

use sonium_xem::schema::{multiphysics_problem_json_schema_string, room_config_json_schema_string};
use std::{env, fs, path::PathBuf};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let output = env::args_os().nth(1).map(PathBuf::from);
    let directory = output.unwrap_or_else(|| PathBuf::from("schemas"));
    fs::create_dir_all(&directory)?;
    fs::write(
        directory.join("room-config-v1.schema.json"),
        format!("{}\n", room_config_json_schema_string()),
    )?;
    fs::write(
        directory.join("multiphysics-problem-v1.schema.json"),
        format!("{}\n", multiphysics_problem_json_schema_string()),
    )?;
    println!("exported XEM schemas to {}", directory.display());
    Ok(())
}
