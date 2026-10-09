//! CSV export for simulation results
//!
//! Converts complex pressure data to RoomEQ's extended CSV measurement contract.

use anyhow::{Result, bail};
use std::io::{BufWriter, Write};
use std::path::Path;

use crate::{SimulationOutput, room_metrics::peak_pressure_to_spl};

/// Deterministic frequency-domain simulations contain no uncorrelated sweep
/// noise, so unity coherence expresses the actual synthetic data model.
const SIMULATED_COHERENCE: f64 = 1.0;

/// Finite sentinel for a noiseless simulation. This is deliberately not
/// presented as an estimate of measured ambient or acquisition noise.
const SIMULATED_NOISE_FLOOR_DB: f64 = -120.0;

/// Export simulation output to CSV files.
///
/// Creates one CSV per (source, listening position) pair with columns:
///   freq,spl,phase,coherence,noise_floor_db
///
/// Files are named `{source_name}_lp{lp_idx}.csv` under `output_dir`.
pub fn export_csvs(output: &SimulationOutput, output_dir: &Path) -> Result<Vec<String>> {
    if output.frequencies.is_empty() {
        bail!("cannot export an empty frequency grid");
    }
    for (index, pair) in output.frequencies.windows(2).enumerate() {
        if !pair[0].is_finite() || !pair[1].is_finite() || pair[0] >= pair[1] {
            bail!(
                "frequency grid must be finite and strictly increasing; indices {} and {} contain {} then {}",
                index,
                index + 1,
                pair[0],
                pair[1]
            );
        }
    }
    if output.frequencies[0] <= 0.0 {
        bail!("frequency grid must contain only positive frequencies");
    }
    output
        .validate_layout()
        .map_err(|error| anyhow::anyhow!(error))?;

    std::fs::create_dir_all(output_dir)?;

    let mut files_written = Vec::new();

    // Provenance sidecar: one row per frequency telling consumers which
    // samples are physically solved and which are reduced-order/synthetic.
    // Kept out of the per-LP CSVs so the 5-column RoomEQ contract is
    // untouched.
    {
        let filepath = output_dir.join("provenance.csv");
        let file = std::fs::File::create(&filepath)?;
        let mut writer = BufWriter::new(file);
        writeln!(writer, "freq,provenance")?;
        for (freq, provenance) in output.frequencies.iter().zip(output.provenance.iter()) {
            writeln!(writer, "{:.4},{}", freq, provenance.as_str())?;
        }
        writer.flush()?;
        files_written.push("provenance.csv".to_string());
    }

    for (src_idx, source_name) in output.source_names.iter().enumerate() {
        let n_lps = output.pressures[src_idx].len();
        for lp_idx in 0..n_lps {
            let filename = format!("{}_lp{}.csv", source_name, lp_idx);
            let filepath = output_dir.join(&filename);

            // BufWriter coalesces the per-row writes into 8 KiB blocks, which
            // for typical 500-row sweeps cuts the syscall count from ~500 to
            // ~3 and makes large-dataset exports an order of magnitude faster.
            let file = std::fs::File::create(&filepath)?;
            let mut writer = BufWriter::new(file);
            writeln!(writer, "freq,spl,phase,coherence,noise_floor_db")?;

            for (freq_idx, &freq) in output.frequencies.iter().enumerate() {
                let pressure = output.pressures[src_idx][lp_idx][freq_idx];
                let spl = peak_pressure_to_spl(pressure);
                let phase = pressure.arg().to_degrees();

                let spl_clamped = if spl.is_finite() {
                    spl.clamp(-120.0, 150.0)
                } else {
                    -120.0
                };
                let phase_clean = if phase.is_finite() { phase } else { 0.0 };

                writeln!(
                    writer,
                    "{:.4},{:.4},{:.4},{:.4},{:.4}",
                    freq, spl_clamped, phase_clean, SIMULATED_COHERENCE, SIMULATED_NOISE_FLOOR_DB,
                )?;
            }
            // Flush before drop so any IO error surfaces as a real error,
            // not a silent failure from BufWriter's drop impl.
            writer.flush()?;

            files_written.push(filename);
        }
    }

    Ok(files_written)
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_complex::Complex64;

    #[test]
    fn rejects_non_monotonic_frequency_grid_before_writing_files() {
        let output = SimulationOutput::solved(
            vec![20.0, 20_200.0, 20_197.9698],
            vec![vec![vec![Complex64::new(1.0, 0.0); 3]]],
            vec!["left".to_string()],
        );
        let directory = tempfile::TempDir::new().unwrap();

        let error = export_csvs(&output, directory.path()).unwrap_err();

        assert!(error.to_string().contains("strictly increasing"));
        assert!(
            std::fs::read_dir(directory.path())
                .unwrap()
                .next()
                .is_none()
        );
    }

    #[test]
    fn export_writes_provenance_sidecar_aligned_with_frequency_grid() {
        let frequencies = vec![20.0, 100.0, 500.0];
        let mut output = SimulationOutput::solved(
            frequencies.clone(),
            vec![vec![vec![Complex64::new(1.0, 0.0); 3]]],
            vec!["left".to_string()],
        );
        output.provenance[2] = crate::FrequencyProvenance::Synthetic;
        let directory = tempfile::TempDir::new().unwrap();

        let files = export_csvs(&output, directory.path()).unwrap();

        assert!(files.iter().any(|name| name == "provenance.csv"));
        let contents = std::fs::read_to_string(directory.path().join("provenance.csv")).unwrap();
        let lines: Vec<&str> = contents.lines().collect();
        assert_eq!(lines[0], "freq,provenance");
        assert_eq!(lines.len(), 1 + frequencies.len());
        assert!(lines[1].ends_with(",solved"));
        assert!(lines[3].ends_with(",synthetic"));
    }
}
