//! Shallow-water material/topography parameters.

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ShallowWaterMaterial {
    pub gravity: f64,
    pub mean_depth: f64,
    pub manning_n: f64,
}

impl ShallowWaterMaterial {
    pub fn wave_speed(&self) -> f64 {
        (self.gravity * self.mean_depth).sqrt()
    }

    pub fn manning_friction_slope(&self, velocity_magnitude: f64) -> f64 {
        self.manning_n * self.manning_n * velocity_magnitude * velocity_magnitude
            / self.mean_depth.powf(4.0 / 3.0)
    }

    pub fn froude_number(&self, velocity_magnitude: f64) -> f64 {
        velocity_magnitude / self.wave_speed().max(1.0e-30)
    }
}

pub fn lake_at_rest_free_surface(depth: f64, bed_elevation: f64) -> f64 {
    depth + bed_elevation
}

pub fn dam_break_dry_bed_front_speed(gravity: f64, left_depth: f64) -> f64 {
    2.0 * (gravity * left_depth).sqrt()
}

pub fn shallow_water_hll_wave_speeds(
    gravity: f64,
    left_depth: f64,
    left_velocity: f64,
    right_depth: f64,
    right_velocity: f64,
) -> (f64, f64) {
    let cl = (gravity * left_depth.max(0.0)).sqrt();
    let cr = (gravity * right_depth.max(0.0)).sqrt();
    (
        (left_velocity - cl).min(right_velocity - cr),
        (left_velocity + cl).max(right_velocity + cr),
    )
}
