//! CEA-2034 measurement generation and file output.
//!
//! Generates 72 evaluation points (36 horizontal + 36 vertical at 1 m)
//! and writes per-angle measurement files to a `cea2034/` subdirectory.
//!
//! Also provides spinorama scoring via `sonium_roomeq::cea2034`.

use std::io::Write as IoWrite;
use std::path::Path;

use super::simulation::SimulationResult;

/// Number of angles per plane (every 10° from 0° to 350°).
pub const ANGLES_PER_PLANE: usize = 36;

/// Total number of evaluation points (horizontal + vertical).
pub const TOTAL_EVAL_POINTS: usize = 2 * ANGLES_PER_PLANE;

/// Generate the 72 CEA-2034 evaluation points at 1 m from the baffle center.
///
/// Returns an array of 72 points: first 36 are horizontal plane (rotating
/// around y-axis), next 36 are vertical plane (rotating around x-axis).
///
/// Coordinate system: z = acoustic axis (forward), y = up, x = right.
pub fn generate_eval_points(baffle_center: [f64; 3]) -> Vec<[f64; 3]> {
    let mut points = Vec::with_capacity(TOTAL_EVAL_POINTS);

    // Horizontal plane: rotate around y-axis
    for i in 0..ANGLES_PER_PLANE {
        let angle_rad = (i as f64 * 10.0).to_radians();
        points.push([
            baffle_center[0] + angle_rad.sin(),
            baffle_center[1],
            baffle_center[2] + angle_rad.cos(),
        ]);
    }

    // Vertical plane: rotate around x-axis
    for i in 0..ANGLES_PER_PLANE {
        let angle_rad = (i as f64 * 10.0).to_radians();
        points.push([
            baffle_center[0],
            baffle_center[1] + angle_rad.sin(),
            baffle_center[2] + angle_rad.cos(),
        ]);
    }

    points
}

/// Write CEA-2034 measurement files.
///
/// Creates `{output_dir}/cea2034/H {angle}.txt` and `V {angle}.txt` files,
/// each containing `freq_hz  spl_db  phase_deg` rows.
pub fn write_cea2034_files(
    output_dir: &Path,
    result: &SimulationResult,
) -> std::io::Result<()> {
    let cea_dir = output_dir.join("cea2034");
    std::fs::create_dir_all(&cea_dir)?;

    let n_freqs = result.frequencies.len();

    // Horizontal files: H 0.txt, H 10.txt, ..., H 350.txt
    for i in 0..ANGLES_PER_PLANE {
        let angle = i * 10;
        let path = cea_dir.join(format!("H {angle}.txt"));
        write_angle_file(&path, &result.frequencies, n_freqs, i, &result.spl, &result.phase)?;
    }

    // Vertical files: V 0.txt, V 10.txt, ..., V 350.txt
    for i in 0..ANGLES_PER_PLANE {
        let angle = i * 10;
        let path = cea_dir.join(format!("V {angle}.txt"));
        let point_idx = ANGLES_PER_PLANE + i;
        write_angle_file(
            &path,
            &result.frequencies,
            n_freqs,
            point_idx,
            &result.spl,
            &result.phase,
        )?;
    }

    Ok(())
}

fn write_angle_file(
    path: &Path,
    frequencies: &[f64],
    n_freqs: usize,
    point_idx: usize,
    spl: &[Vec<f64>],
    phase: &[Vec<f64>],
) -> std::io::Result<()> {
    let mut f = std::fs::File::create(path)?;
    for fi in 0..n_freqs {
        let freq = frequencies[fi];
        let s = spl[fi][point_idx];
        let p_deg = phase[fi][point_idx].to_degrees();
        writeln!(f, "{freq:.2}\t{s:.4}\t{p_deg:.4}")?;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Spinorama scoring
// ---------------------------------------------------------------------------

/// Spinorama-derived metrics from a simulation result.
#[cfg(feature = "cli")]
pub struct SpinoramaMetrics {
    /// On-axis SPL curve (frequencies × SPL dB).
    pub on_axis: Vec<f64>,
    /// Listening Window SPL.
    pub listening_window: Vec<f64>,
    /// Early Reflections SPL.
    pub early_reflections: Vec<f64>,
    /// Sound Power SPL.
    pub sound_power: Vec<f64>,
    /// Predicted In-Room response.
    pub pir: Vec<f64>,
    /// CEA-2034 preference score.
    pub preference_score: f64,
}

/// Compute spinorama metrics from a simulation result.
#[cfg(feature = "cli")]
pub fn compute_spinorama_metrics(result: &SimulationResult) -> SpinoramaMetrics {
    let n_freqs = result.frequencies.len();

    let mut spl_h = vec![vec![0.0f64; n_freqs]; ANGLES_PER_PLANE];
    let mut spl_v = vec![vec![0.0f64; n_freqs]; ANGLES_PER_PLANE];

    for i in 0..ANGLES_PER_PLANE {
        let angle_deg = i as i32 * 10;
        let signed = if angle_deg <= 180 {
            angle_deg
        } else {
            angle_deg - 360
        };
        let row = ((signed + 170) / 10) as usize;

        for fi in 0..n_freqs {
            spl_h[row][fi] = result.spl[fi][i];
            spl_v[row][fi] = result.spl[fi][ANGLES_PER_PLANE + i];
        }
    }

    let freq_arr = result.frequencies.clone();

    let on_axis: Vec<f64> = (0..n_freqs).map(|fi| spl_h[17][fi]).collect();

    let lw_rows_h = [16, 18, 15, 19, 14, 20];
    let lw_rows_v = [16, 18, 15, 19, 14, 20];
    let lw = rms_average_hv(&spl_h, &spl_v, &lw_rows_h, &lw_rows_v, n_freqs);

    let er_rows_h = [13, 21, 12, 22, 11, 23];
    let er_rows_v = [13, 21, 12, 22, 11, 23];
    let er = rms_average_hv(&spl_h, &spl_v, &er_rows_h, &er_rows_v, n_freqs);

    let sp = weighted_rms_all(&spl_h, &spl_v, n_freqs);

    let pir = sonium_roomeq::cea2034::compute_pir_from_lw_er_sp(&lw, &er, &sp);

    let intervals = sonium_roomeq::cea2034::octave_intervals(2, &freq_arr);
    let metrics = sonium_roomeq::cea2034::score(&freq_arr, &intervals, &on_axis, &lw, &sp, &pir);

    SpinoramaMetrics {
        on_axis,
        listening_window: lw,
        early_reflections: er,
        sound_power: sp,
        pir,
        preference_score: metrics.pref_score,
    }
}

/// RMS average of selected rows from H and V SPL matrices.
#[cfg(feature = "cli")]
fn rms_average_hv(
    spl_h: &[Vec<f64>],
    spl_v: &[Vec<f64>],
    rows_h: &[usize],
    rows_v: &[usize],
    n_freqs: usize,
) -> Vec<f64> {
    let total = rows_h.len() + rows_v.len();
    (0..n_freqs)
        .map(|fi| {
            let mut sum_p2 = 0.0;
            for &r in rows_h {
                let p = 10f64.powf((spl_h[r][fi] - 105.0) / 20.0);
                sum_p2 += p * p;
            }
            for &r in rows_v {
                let p = 10f64.powf((spl_v[r][fi] - 105.0) / 20.0);
                sum_p2 += p * p;
            }
            let rms_p = (sum_p2 / total as f64).sqrt();
            20.0 * rms_p.log10() + 105.0
        })
        .collect()
}

/// Solid-angle-weighted RMS over all rows from H and V SPL matrices.
#[cfg(feature = "cli")]
fn weighted_rms_all(spl_h: &[Vec<f64>], spl_v: &[Vec<f64>], n_freqs: usize) -> Vec<f64> {
    let mut weights = Vec::with_capacity(ANGLES_PER_PLANE);
    for r in 0..ANGLES_PER_PLANE {
        let angle_deg = r as f64 * 10.0 - 170.0;
        let w = angle_deg.to_radians().cos().abs().max(0.01);
        weights.push(w);
    }
    let total_w: f64 = weights.iter().sum::<f64>() * 2.0;

    (0..n_freqs)
        .map(|fi| {
            let mut sum_wp2 = 0.0;
            for r in 0..ANGLES_PER_PLANE {
                let p_h = 10f64.powf((spl_h[r][fi] - 105.0) / 20.0);
                let p_v = 10f64.powf((spl_v[r][fi] - 105.0) / 20.0);
                sum_wp2 += weights[r] * (p_h * p_h + p_v * p_v);
            }
            let rms_p = (sum_wp2 / total_w).sqrt();
            20.0 * rms_p.log10() + 105.0
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_eval_points_count() {
        let pts = generate_eval_points([0.0, 0.0, 0.14]);
        assert_eq!(pts.len(), TOTAL_EVAL_POINTS);
    }

    #[test]
    fn test_eval_points_at_1m() {
        let center = [0.0, 0.0, 0.14];
        let pts = generate_eval_points(center);
        for pt in &pts {
            let dx = pt[0] - center[0];
            let dy = pt[1] - center[1];
            let dz = pt[2] - center[2];
            let dist = (dx * dx + dy * dy + dz * dz).sqrt();
            assert!(
                (dist - 1.0).abs() < 1e-10,
                "point distance from center = {dist}"
            );
        }
    }

    #[test]
    fn test_on_axis_is_first_point() {
        let center = [0.0, 0.0, 0.14];
        let pts = generate_eval_points(center);
        assert!((pts[0][0] - center[0]).abs() < 1e-10);
        assert!((pts[0][1] - center[1]).abs() < 1e-10);
        assert!((pts[0][2] - center[2] - 1.0).abs() < 1e-10);
        assert!((pts[36][0] - center[0]).abs() < 1e-10);
        assert!((pts[36][1] - center[1]).abs() < 1e-10);
        assert!((pts[36][2] - center[2] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_write_cea2034_files() {
        let result = SimulationResult {
            frequencies: vec![100.0, 1000.0, 10000.0],
            spl: vec![
                vec![80.0; TOTAL_EVAL_POINTS],
                vec![85.0; TOTAL_EVAL_POINTS],
                vec![75.0; TOTAL_EVAL_POINTS],
            ],
            phase: vec![
                vec![0.0; TOTAL_EVAL_POINTS],
                vec![-0.5; TOTAL_EVAL_POINTS],
                vec![-1.0; TOTAL_EVAL_POINTS],
            ],
        };

        let dir = std::env::temp_dir().join("speaker_cea2034_test");
        let _ = std::fs::remove_dir_all(&dir);
        write_cea2034_files(&dir, &result).unwrap();

        let cea_dir = dir.join("cea2034");
        assert!(cea_dir.join("H 0.txt").exists());
        assert!(cea_dir.join("H 350.txt").exists());
        assert!(cea_dir.join("V 0.txt").exists());
        assert!(cea_dir.join("V 350.txt").exists());

        let content = std::fs::read_to_string(cea_dir.join("H 0.txt")).unwrap();
        let lines: Vec<&str> = content.trim().lines().collect();
        assert_eq!(lines.len(), 3);
        assert!(lines[0].starts_with("100.00"));
        assert!(lines[0].contains("80.0"));

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_angle_mapping_row17_is_on_axis() {
        let angle_deg = 0_i32;
        let signed = angle_deg;
        let row = (signed + 170) / 10;
        assert_eq!(row, 17);
    }

    #[test]
    fn test_angle_mapping_coverage() {
        let mut rows = Vec::new();
        for i in 0..36 {
            let angle_deg = i as i32 * 10;
            let signed = if angle_deg <= 180 {
                angle_deg
            } else {
                angle_deg - 360
            };
            let row = ((signed + 170) / 10) as usize;
            assert!(row < 36, "row {row} out of range for angle {angle_deg}°");
            rows.push(row);
        }
        rows.sort();
        rows.dedup();
        assert_eq!(rows.len(), 36, "not all rows covered: {rows:?}");
    }
}
