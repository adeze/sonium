//! Fisher-KPP, Burke-Schumann, and Frank-Kamenetskii references.

pub fn fisher_kpp_minimum_wave_speed(diffusivity: f64, reaction_rate: f64) -> f64 {
    assert!(diffusivity > 0.0 && reaction_rate >= 0.0);
    2.0 * (diffusivity * reaction_rate).sqrt()
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BurkeSchumannState {
    pub mixture_fraction: f64,
    pub fuel: f64,
    pub oxidizer: f64,
    pub temperature: f64,
}

/// One-dimensional flame-sheet solution with `Z=1-x/L`.
pub fn burke_schumann_flame_sheet(
    x: f64,
    length: f64,
    stoichiometric_mixture_fraction: f64,
    unburned_temperature: f64,
    adiabatic_flame_temperature: f64,
) -> BurkeSchumannState {
    assert!(length > 0.0 && (0.0..1.0).contains(&stoichiometric_mixture_fraction));
    let mixture_fraction = 1.0 - x.clamp(0.0, length) / length;
    let z_st = stoichiometric_mixture_fraction;
    let (fuel, oxidizer, progress) = if mixture_fraction >= z_st {
        (
            (mixture_fraction - z_st) / (1.0 - z_st),
            0.0,
            (1.0 - mixture_fraction) / (1.0 - z_st),
        )
    } else {
        (
            0.0,
            (z_st - mixture_fraction) / z_st,
            mixture_fraction / z_st,
        )
    };
    BurkeSchumannState {
        mixture_fraction,
        fuel,
        oxidizer,
        temperature: unburned_temperature
            + progress * (adiabatic_flame_temperature - unburned_temperature),
    }
}

pub fn burke_schumann_flame_location(length: f64, stoichiometric_mixture_fraction: f64) -> f64 {
    length * (1.0 - stoichiometric_mixture_fraction)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FrankKamenetskiiGeometry {
    PlaneSlab,
    InfiniteCylinder,
    Sphere,
}

pub fn frank_kamenetskii_critical_parameter(geometry: FrankKamenetskiiGeometry) -> f64 {
    match geometry {
        FrankKamenetskiiGeometry::PlaneSlab => 0.878,
        FrankKamenetskiiGeometry::InfiniteCylinder => 2.0,
        FrankKamenetskiiGeometry::Sphere => 3.32,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fisher_front_speed_has_sqrt_diffusion_reaction_scaling() {
        let speed = fisher_kpp_minimum_wave_speed(2.0, 8.0);
        assert_eq!(speed, 8.0);
        assert_eq!(fisher_kpp_minimum_wave_speed(2.0, 0.0), 0.0);
    }

    #[test]
    fn burke_schumann_flame_is_stoichiometric_and_element_bounded() {
        let length = 1.0;
        let z_st = 0.25;
        let x_flame = burke_schumann_flame_location(length, z_st);
        let flame = burke_schumann_flame_sheet(x_flame, length, z_st, 300.0, 2200.0);
        assert!((flame.mixture_fraction - z_st).abs() < 1e-15);
        assert_eq!(flame.fuel, 0.0);
        assert_eq!(flame.oxidizer, 0.0);
        assert_eq!(flame.temperature, 2200.0);
    }

    #[test]
    fn frank_kamenetskii_critical_values_follow_geometry() {
        let slab = frank_kamenetskii_critical_parameter(FrankKamenetskiiGeometry::PlaneSlab);
        let cylinder =
            frank_kamenetskii_critical_parameter(FrankKamenetskiiGeometry::InfiniteCylinder);
        let sphere = frank_kamenetskii_critical_parameter(FrankKamenetskiiGeometry::Sphere);
        assert!(slab < cylinder && cylinder < sphere);
        assert_eq!((slab, cylinder, sphere), (0.878, 2.0, 3.32));
    }
}
