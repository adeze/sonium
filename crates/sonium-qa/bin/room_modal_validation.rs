//! Analytical-reference check of rigid rectangular-room modal frequencies.
//!
//! SCOPE: analytical reference only. This binary verifies the closed-form
//! Neumann eigenfrequency formula of a shoebox room, the axial/tangential/
//! oblique family classification, mode ordering, and cube degeneracy:
//!
//!     f_mnp = c / 2 * sqrt((m / Lx)^2 + (n / Ly)^2 + (p / Lz)^2)
//!
//! It assembles no FEM operators and solves no numerical eigenproblem, so it
//! is not evidence of Sonium solver accuracy. Numerical room eigenmodes
//! (solved frequencies, eigen-residuals, modal fields) are covered by
//! `qa-room-shoebox-fem-benchmark`; do not cite this binary for that claim.

// Named acceptance tolerances for the analytical reference checks below.
const MODE_FREQUENCY_TOLERANCE_HZ: f64 = 5.0e-7;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ModeIndex {
    nx: u32,
    ny: u32,
    nz: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ModeFamily {
    Axial,
    Tangential,
    Oblique,
}

#[derive(Clone, Copy, Debug)]
struct RoomMode {
    index: ModeIndex,
    frequency_hz: f64,
    family: ModeFamily,
}

#[derive(Clone, Copy, Debug)]
struct RectangularRoomModes {
    width: f64,
    depth: f64,
    height: f64,
    sound_speed: f64,
}

impl RectangularRoomModes {
    fn new(width: f64, depth: f64, height: f64, sound_speed: f64) -> Result<Self, String> {
        for (name, value) in [
            ("width", width),
            ("depth", depth),
            ("height", height),
            ("sound_speed", sound_speed),
        ] {
            if !value.is_finite() || value <= 0.0 {
                return Err(format!("{name} must be finite and positive, got {value}"));
            }
        }

        Ok(Self {
            width,
            depth,
            height,
            sound_speed,
        })
    }

    fn frequency(&self, index: ModeIndex) -> f64 {
        let nx = index.nx as f64 / self.width;
        let ny = index.ny as f64 / self.depth;
        let nz = index.nz as f64 / self.height;
        0.5 * self.sound_speed * (nx * nx + ny * ny + nz * nz).sqrt()
    }

    fn mode(&self, index: ModeIndex) -> RoomMode {
        RoomMode {
            index,
            frequency_hz: self.frequency(index),
            family: classify_mode(index),
        }
    }

    fn modes_through(&self, max_index: u32) -> Vec<RoomMode> {
        let mut modes = Vec::new();

        for nx in 0..=max_index {
            for ny in 0..=max_index {
                for nz in 0..=max_index {
                    if nx == 0 && ny == 0 && nz == 0 {
                        continue;
                    }
                    modes.push(self.mode(ModeIndex { nx, ny, nz }));
                }
            }
        }

        modes.sort_by(|a, b| {
            a.frequency_hz
                .total_cmp(&b.frequency_hz)
                .then_with(|| a.index.nx.cmp(&b.index.nx))
                .then_with(|| a.index.ny.cmp(&b.index.ny))
                .then_with(|| a.index.nz.cmp(&b.index.nz))
        });
        modes
    }
}

fn classify_mode(index: ModeIndex) -> ModeFamily {
    match [index.nx, index.ny, index.nz]
        .into_iter()
        .filter(|n| *n > 0)
        .count()
    {
        1 => ModeFamily::Axial,
        2 => ModeFamily::Tangential,
        3 => ModeFamily::Oblique,
        _ => panic!("zero mode has no acoustic modal family"),
    }
}

fn assert_close(label: &str, actual: f64, expected: f64, tolerance: f64) -> Result<(), String> {
    let error = (actual - expected).abs();
    if error > tolerance {
        return Err(format!(
            "{label}: expected {expected:.9} Hz, got {actual:.9} Hz (abs error {error:.3e})"
        ));
    }
    Ok(())
}

fn assert_mode(
    room: &RectangularRoomModes,
    index: ModeIndex,
    expected_hz: f64,
    expected_family: ModeFamily,
) -> Result<(), String> {
    let mode = room.mode(index);
    assert_close(
        &format!("mode ({}, {}, {})", index.nx, index.ny, index.nz),
        mode.frequency_hz,
        expected_hz,
        MODE_FREQUENCY_TOLERANCE_HZ,
    )?;

    if mode.family != expected_family {
        return Err(format!(
            "mode ({}, {}, {}) classified as {:?}, expected {:?}",
            index.nx, index.ny, index.nz, mode.family, expected_family
        ));
    }

    println!(
        "  ({}, {}, {}) {:?}: {:.6} Hz",
        index.nx, index.ny, index.nz, mode.family, mode.frequency_hz
    );
    Ok(())
}

fn validate_medium_room_modes() -> Result<(), String> {
    let room = RectangularRoomModes::new(5.0, 4.0, 2.5, 343.0)?;

    println!("Medium rigid shoebox room, 5.0 x 4.0 x 2.5 m:");
    assert_mode(
        &room,
        ModeIndex {
            nx: 1,
            ny: 0,
            nz: 0,
        },
        34.3,
        ModeFamily::Axial,
    )?;
    assert_mode(
        &room,
        ModeIndex {
            nx: 0,
            ny: 1,
            nz: 0,
        },
        42.875,
        ModeFamily::Axial,
    )?;
    assert_mode(
        &room,
        ModeIndex {
            nx: 1,
            ny: 1,
            nz: 0,
        },
        54.9067903360,
        ModeFamily::Tangential,
    )?;
    assert_mode(
        &room,
        ModeIndex {
            nx: 0,
            ny: 0,
            nz: 1,
        },
        68.6,
        ModeFamily::Axial,
    )?;
    assert_mode(
        &room,
        ModeIndex {
            nx: 1,
            ny: 1,
            nz: 1,
        },
        87.8676028181,
        ModeFamily::Oblique,
    )?;

    let first_modes = room.modes_through(2);
    let expected_order = [
        ModeIndex {
            nx: 1,
            ny: 0,
            nz: 0,
        },
        ModeIndex {
            nx: 0,
            ny: 1,
            nz: 0,
        },
        ModeIndex {
            nx: 1,
            ny: 1,
            nz: 0,
        },
        ModeIndex {
            nx: 0,
            ny: 0,
            nz: 1,
        },
    ];

    for (i, expected) in expected_order.into_iter().enumerate() {
        let actual = first_modes[i].index;
        if actual != expected {
            return Err(format!(
                "mode ordering mismatch at rank {}: expected {:?}, got {:?}",
                i + 1,
                expected,
                actual
            ));
        }
    }

    Ok(())
}

fn validate_cube_degeneracy() -> Result<(), String> {
    let room = RectangularRoomModes::new(3.0, 3.0, 3.0, 343.0)?;

    let axial = [
        room.frequency(ModeIndex {
            nx: 1,
            ny: 0,
            nz: 0,
        }),
        room.frequency(ModeIndex {
            nx: 0,
            ny: 1,
            nz: 0,
        }),
        room.frequency(ModeIndex {
            nx: 0,
            ny: 0,
            nz: 1,
        }),
    ];

    println!("Cube degeneracy, 3.0 x 3.0 x 3.0 m:");
    for (i, frequency) in axial.into_iter().enumerate() {
        println!("  axial mode {}: {:.6} Hz", i + 1, frequency);
        assert_close(
            "cube axial degeneracy",
            frequency,
            57.1666666667,
            MODE_FREQUENCY_TOLERANCE_HZ,
        )?;
    }

    let tangential = room.frequency(ModeIndex {
        nx: 1,
        ny: 1,
        nz: 0,
    });
    assert_close(
        "cube tangential mode",
        tangential,
        80.8458753157,
        MODE_FREQUENCY_TOLERANCE_HZ,
    )?;
    Ok(())
}

fn run() -> Result<(), String> {
    println!("Room acoustics modal validation:");
    validate_medium_room_modes()?;
    validate_cube_degeneracy()?;
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("FAILED: {error}");
        println!("FAILED");
        std::process::exit(1);
    }

    println!("SUCCESS");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classifies_axial_tangential_and_oblique_modes() {
        assert_eq!(
            classify_mode(ModeIndex {
                nx: 1,
                ny: 0,
                nz: 0
            }),
            ModeFamily::Axial
        );
        assert_eq!(
            classify_mode(ModeIndex {
                nx: 1,
                ny: 1,
                nz: 0
            }),
            ModeFamily::Tangential
        );
        assert_eq!(
            classify_mode(ModeIndex {
                nx: 1,
                ny: 1,
                nz: 1
            }),
            ModeFamily::Oblique
        );
    }

    #[test]
    fn cube_modes_are_degenerate() {
        validate_cube_degeneracy().expect("cube degeneracy reference should hold");
    }

    #[test]
    fn medium_room_reference_modes_match() {
        validate_medium_room_modes().expect("medium room modal references should hold");
    }
}
