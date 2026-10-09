//! Magnetic material properties

/// Permeability of free space (H/m).
pub const MU_0: f64 = 4.0 * std::f64::consts::PI * 1e-7;

/// Magnetic material region.
#[derive(Debug, Clone)]
pub enum MagneticMaterial {
    /// Air or non-magnetic region: mu = mu_0
    Air,
    /// Linear soft magnetic material (iron, steel): mu = mu_r * mu_0
    Linear { mu_r: f64 },
    /// Permanent magnet: mu = mu_r * mu_0 with remanent flux density B_r
    PermanentMagnet {
        mu_r: f64,
        /// Remanent flux density magnitude (T). Typical NdFeB: 1.2-1.4 T.
        br_magnitude: f64,
        /// Magnetization direction (unit vector) [x, y, z].
        direction: [f64; 3],
    },
}

impl MagneticMaterial {
    /// Air.
    pub fn air() -> Self {
        Self::Air
    }

    /// Soft iron with given relative permeability.
    pub fn iron(mu_r: f64) -> Self {
        Self::Linear { mu_r }
    }

    /// NdFeB permanent magnet with given Br and magnetization direction.
    pub fn neodymium(br: f64, direction: [f64; 3]) -> Self {
        // Typical NdFeB: mu_r ~= 1.05
        Self::PermanentMagnet {
            mu_r: 1.05,
            br_magnitude: br,
            direction: normalize(direction),
        }
    }

    /// Ferrite permanent magnet.
    pub fn ferrite(br: f64, direction: [f64; 3]) -> Self {
        Self::PermanentMagnet {
            mu_r: 1.1,
            br_magnitude: br,
            direction: normalize(direction),
        }
    }

    /// Absolute permeability mu = mu_r * mu_0.
    pub fn permeability(&self) -> f64 {
        match self {
            Self::Air => MU_0,
            Self::Linear { mu_r } => mu_r * MU_0,
            Self::PermanentMagnet { mu_r, .. } => mu_r * MU_0,
        }
    }

    /// Remanent flux density vector B_r (zero for non-magnet materials).
    pub fn remanent_b(&self) -> [f64; 3] {
        match self {
            Self::Air | Self::Linear { .. } => [0.0, 0.0, 0.0],
            Self::PermanentMagnet {
                br_magnitude,
                direction,
                ..
            } => [
                br_magnitude * direction[0],
                br_magnitude * direction[1],
                br_magnitude * direction[2],
            ],
        }
    }
}

fn normalize(v: [f64; 3]) -> [f64; 3] {
    let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    assert!(len > 1e-15, "Direction vector must be non-zero");
    [v[0] / len, v[1] / len, v[2] / len]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_air_permeability() {
        let mat = MagneticMaterial::air();
        assert!((mat.permeability() - MU_0).abs() < 1e-20);
        assert_eq!(mat.remanent_b(), [0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_neodymium_magnet() {
        let mat = MagneticMaterial::neodymium(1.3, [0.0, 0.0, 1.0]);
        let br = mat.remanent_b();
        assert!(br[0].abs() < 1e-14);
        assert!(br[1].abs() < 1e-14);
        assert!((br[2] - 1.3).abs() < 1e-14);
    }
}
