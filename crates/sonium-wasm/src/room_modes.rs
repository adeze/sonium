//! Room mode calculations

use num_complex::Complex64;
use std::f64::consts::PI;

// Re-export types from parent
pub use crate::{Point3D, RoomMode};

/// Calculate room modes for a rectangular room
pub fn calculate_room_modes(
    length_x: f64,
    length_y: f64,
    length_z: f64,
    speed_of_sound: f64,
    max_frequency: f64,
    max_order: u32,
) -> Vec<RoomMode> {
    if !length_x.is_finite()
        || !length_y.is_finite()
        || !length_z.is_finite()
        || !speed_of_sound.is_finite()
        || !max_frequency.is_finite()
        || length_x <= 0.0
        || length_y <= 0.0
        || length_z <= 0.0
        || speed_of_sound <= 0.0
        || max_frequency < 0.0
    {
        return Vec::new();
    }
    let mut modes = Vec::new();
    let c = speed_of_sound;

    for n in 0..=max_order {
        for m in 0..=max_order {
            for p in 0..=max_order {
                if n == 0 && m == 0 && p == 0 {
                    continue;
                }

                let nx = n as f64 / length_x;
                let my = m as f64 / length_y;
                let pz = p as f64 / length_z;
                let freq = (c / 2.0) * (nx * nx + my * my + pz * pz).sqrt();

                if freq > max_frequency {
                    continue;
                }

                let zero_count = [n, m, p].iter().filter(|&&x| x == 0).count();
                let mode_type = match zero_count {
                    2 => "axial",
                    1 => "tangential",
                    0 => "oblique",
                    _ => continue,
                };

                let description = match (n, m, p) {
                    (n, 0, 0) if n > 0 => format!("{},0,0 - Length mode (X)", n),
                    (0, m, 0) if m > 0 => format!("0,{},0 - Width mode (Y)", m),
                    (0, 0, p) if p > 0 => format!("0,0,{} - Height mode (Z)", p),
                    (n, m, 0) => format!("{},{},0 - Floor tangential", n, m),
                    (n, 0, p) => format!("{},0,{} - Side tangential", n, p),
                    (0, m, p) => format!("0,{},{} - Front tangential", m, p),
                    (n, m, p) => format!("{},{},{} - Oblique", n, m, p),
                };

                modes.push(RoomMode {
                    frequency: freq,
                    indices: [n, m, p],
                    mode_type: mode_type.to_string(),
                    description,
                });
            }
        }
    }

    // `total_cmp` keeps this public helper panic-free even when called
    // directly with malformed floating-point inputs. Runtime configuration
    // validation still rejects non-finite dimensions and speed values before
    // simulation begins.
    modes.sort_by(|a, b| a.frequency.total_cmp(&b.frequency));
    modes
}

/// Generate the rectangular-room image sources used by the native fast-room
/// solver. Each tuple contains the image position, the product of wall
/// reflection coefficients, and the total reflection order.
pub fn rectangular_image_sources(
    source: &Point3D,
    dimensions: [f64; 3],
    reflections: [Complex64; 6],
    max_order: usize,
) -> Vec<(Point3D, Complex64, usize)> {
    fn axis_images(
        source: f64,
        length: f64,
        lower: Complex64,
        upper: Complex64,
        max_order: usize,
    ) -> Vec<(f64, Complex64, usize)> {
        let max_n = max_order as isize;
        let mut result = Vec::with_capacity(4 * max_order + 2);
        for n in -max_n..=max_n {
            let paired_order = 2 * n.unsigned_abs();
            if paired_order <= max_order {
                result.push((
                    2.0 * n as f64 * length + source,
                    (lower * upper).powu(n.unsigned_abs() as u32),
                    paired_order,
                ));
            }
            let lower_count = (n - 1).unsigned_abs();
            let upper_count = n.unsigned_abs();
            let order = lower_count + upper_count;
            if order <= max_order {
                result.push((
                    2.0 * n as f64 * length - source,
                    lower.powu(lower_count as u32) * upper.powu(upper_count as u32),
                    order,
                ));
            }
        }
        result
    }

    let axis = [
        axis_images(
            source.x,
            dimensions[0],
            reflections[0],
            reflections[1],
            max_order,
        ),
        axis_images(
            source.y,
            dimensions[1],
            reflections[2],
            reflections[3],
            max_order,
        ),
        axis_images(
            source.z,
            dimensions[2],
            reflections[4],
            reflections[5],
            max_order,
        ),
    ];
    let mut images = Vec::new();
    for x in &axis[0] {
        for y in &axis[1] {
            if x.2 + y.2 > max_order {
                continue;
            }
            for z in &axis[2] {
                let order = x.2 + y.2 + z.2;
                if order <= max_order {
                    images.push((Point3D::new(x.0, y.0, z.0), x.1 * y.1 * z.1, order));
                }
            }
        }
    }
    images
}

/// Calculate modal pressure at a point
#[allow(clippy::too_many_arguments)]
pub fn calculate_modal_pressure(
    source: &Point3D,
    listener: &Point3D,
    frequency: f64,
    room_width: f64,
    room_depth: f64,
    room_height: f64,
    speed_of_sound: f64,
    max_mode_order: u32,
    modal_damping: f64,
) -> Complex64 {
    let volume = room_width * room_depth * room_height;
    let omega = 2.0 * PI * frequency;
    let omega_sq = omega * omega;
    let c_sq = speed_of_sound * speed_of_sound;
    let prefactor = c_sq / volume;
    let mut modal_sum = Complex64::new(0.0, 0.0);

    for n in 0..=max_mode_order {
        for m in 0..=max_mode_order {
            for p in 0..=max_mode_order {
                let nx = n as f64 / room_width;
                let my = m as f64 / room_depth;
                let pz = p as f64 / room_height;
                let omega_n = speed_of_sound * PI * (nx * nx + my * my + pz * pz).sqrt();
                let omega_n_sq = omega_n * omega_n;
                let mode_freq = omega_n / (2.0 * PI);

                // Keep the rigid (0,0,0) mode in the low-frequency response.
                // Filtering it out makes the response spuriously vanish below
                // the first room mode.
                if (n != 0 || m != 0 || p != 0)
                    && (mode_freq > frequency * 4.0 || mode_freq < frequency / 4.0)
                {
                    continue;
                }

                let source_mode = (n as f64 * PI * source.x / room_width).cos()
                    * (m as f64 * PI * source.y / room_depth).cos()
                    * (p as f64 * PI * source.z / room_height).cos();

                let listener_mode = (n as f64 * PI * listener.x / room_width).cos()
                    * (m as f64 * PI * listener.y / room_depth).cos()
                    * (p as f64 * PI * listener.z / room_height).cos();

                let epsilon = |i: u32| if i == 0 { 1.0 } else { 2.0 };
                let mode_norm = epsilon(n) * epsilon(m) * epsilon(p);

                // For u¨ + 2ζω_n u˙ + ω_n²u = f, the e^{-iωt}
                // denominator is ω_n² - ω² - 2iζω_nω.  The previous inverse
                // dependence on ζ made *more* damping produce a sharper pole.
                let delta_n = modal_damping.max(0.0) * omega_n;
                let denominator = Complex64::new(omega_n_sq - omega_sq, -2.0 * delta_n * omega);
                let transfer_function = Complex64::new(1.0, 0.0) / denominator;

                let mode_amplitude = mode_norm * source_mode * listener_mode;
                modal_sum += transfer_function * mode_amplitude;
            }
        }
    }

    modal_sum *= prefactor;
    // The eigenfunctions already describe the standing-wave spatial phase;
    // an additional free-field phase would double-count propagation and break
    // reciprocity of the modal field.
    modal_sum
}

/// Hybrid crossover weight
pub fn hybrid_crossover_weight(
    frequency: f64,
    schroeder_frequency: f64,
    crossover_width_octaves: f64,
) -> f64 {
    if !frequency.is_finite() || !schroeder_frequency.is_finite() {
        return 0.0;
    }
    if schroeder_frequency <= 0.0 {
        return 1.0;
    }

    if crossover_width_octaves <= 0.0 || !crossover_width_octaves.is_finite() {
        return if frequency >= schroeder_frequency {
            1.0
        } else {
            0.0
        };
    }

    let octaves_from_schroeder = (frequency / schroeder_frequency).log2();
    let half_width = crossover_width_octaves / 2.0;

    if octaves_from_schroeder < -half_width {
        0.0
    } else if octaves_from_schroeder > half_width {
        1.0
    } else {
        let t = (octaves_from_schroeder + half_width) / crossover_width_octaves;
        (1.0 - (t * PI).cos()) / 2.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hybrid_crossover_weight_below_band() {
        assert_eq!(hybrid_crossover_weight(100.0, 1000.0, 1.0), 0.0);
    }

    #[test]
    fn room_modes_reject_nonfinite_or_nonpositive_inputs_without_panicking() {
        assert!(calculate_room_modes(f64::NAN, 4.0, 2.8, 343.0, 500.0, 4).is_empty());
        assert!(calculate_room_modes(5.0, 4.0, 2.8, 343.0, -1.0, 4).is_empty());
        assert!(calculate_room_modes(5.0, 4.0, 2.8, 0.0, 500.0, 4).is_empty());
    }

    #[test]
    fn hybrid_crossover_weight_above_band() {
        assert_eq!(hybrid_crossover_weight(2000.0, 1000.0, 1.0), 1.0);
    }

    #[test]
    fn hybrid_crossover_weight_at_center() {
        let w = hybrid_crossover_weight(1000.0, 1000.0, 1.0);
        assert!((w - 0.5).abs() < 1e-10, "expected 0.5 at center, got {}", w);
    }

    #[test]
    fn hybrid_crossover_weight_zero_schroeder() {
        assert_eq!(hybrid_crossover_weight(100.0, 0.0, 1.0), 1.0);
    }

    #[test]
    fn modal_damping_reduces_resonant_pressure() {
        let source = Point3D {
            x: 0.4,
            y: 0.7,
            z: 0.8,
        };
        let listener = Point3D {
            x: 1.1,
            y: 1.3,
            z: 1.0,
        };
        let mode_frequency = 343.0 / (2.0 * 5.0);
        let lightly_damped = calculate_modal_pressure(
            &source,
            &listener,
            mode_frequency,
            5.0,
            4.0,
            2.8,
            343.0,
            4,
            0.005,
        );
        let damped = calculate_modal_pressure(
            &source,
            &listener,
            mode_frequency,
            5.0,
            4.0,
            2.8,
            343.0,
            4,
            0.05,
        );
        assert!(damped.norm() < lightly_damped.norm());
    }

    #[test]
    fn rigid_mode_keeps_low_frequency_response_nonzero() {
        let source = Point3D::new(0.4, 0.7, 0.8);
        let listener = Point3D::new(1.1, 1.3, 1.0);
        let pressure =
            calculate_modal_pressure(&source, &listener, 1.0, 5.0, 4.0, 2.8, 343.0, 0, 0.01);
        assert!(pressure.norm() > 0.0);
    }

    #[test]
    fn modal_field_is_reciprocal_without_free_field_phase() {
        let source = Point3D::new(0.4, 0.7, 0.8);
        let listener = Point3D::new(1.1, 1.3, 1.0);
        let forward =
            calculate_modal_pressure(&source, &listener, 180.0, 5.0, 4.0, 2.8, 343.0, 4, 0.01);
        let reverse =
            calculate_modal_pressure(&listener, &source, 180.0, 5.0, 4.0, 2.8, 343.0, 4, 0.01);
        assert!((forward - reverse).norm() < 1e-12);
    }

    #[test]
    fn zero_crossover_width_is_a_step() {
        assert_eq!(hybrid_crossover_weight(499.0, 500.0, 0.0), 0.0);
        assert_eq!(hybrid_crossover_weight(500.0, 500.0, 0.0), 1.0);
    }

    #[test]
    fn rectangular_images_include_repeated_axis_reflections() {
        let images = rectangular_image_sources(
            &Point3D::new(1.0, 1.5, 0.8),
            [5.0, 4.0, 2.8],
            [
                Complex64::new(0.8, 0.0),
                Complex64::new(0.7, 0.0),
                Complex64::new(0.6, 0.0),
                Complex64::new(0.5, 0.0),
                Complex64::new(0.9, 0.0),
                Complex64::new(0.4, 0.0),
            ],
            2,
        );
        assert!(images.iter().any(|(point, coefficient, order)| {
            *order == 2
                && (point.x - 11.0).abs() < 1e-12
                && (point.y - 1.5).abs() < 1e-12
                && (point.z - 0.8).abs() < 1e-12
                && (coefficient.re - 0.8 * 0.7).abs() < 1e-12
        }));
    }

    #[test]
    fn hybrid_crossover_weight_monotonic() {
        let fs: Vec<f64> = (1..=100).map(|i| 100.0 + i as f64 * 10.0).collect();
        let schroeder = 500.0;
        let width = 2.0;
        for i in 1..fs.len() {
            let w0 = hybrid_crossover_weight(fs[i - 1], schroeder, width);
            let w1 = hybrid_crossover_weight(fs[i], schroeder, width);
            assert!(
                w1 >= w0 - 1e-12,
                "crossover weight not monotonic at {} Hz: {} -> {}",
                fs[i],
                w0,
                w1
            );
        }
    }
}
