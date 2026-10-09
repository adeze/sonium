//! Public-API verification for the analytical cases in 20260806-multiphysics.md.

use std::f64::consts::PI;

use num_complex::Complex64;
use sonium_fem::physics::{
    coupled::battery::{
        de_levie_impedance, sand_depletion_time, spherical_particle_average_concentration,
        spherical_particle_surface_concentration,
    },
    coupled::conjugate_heat_transfer::{
        GraetzBoundaryCondition, adiabatic_fin_base_heat_rate, adiabatic_fin_temperature_excess,
        fully_developed_graetz_nusselt, graetz_bulk_temperature, plane_wall_solution,
    },
    coupled::electrothermal::{
        dc_heated_bar, dodd_deeds_filament_impedance_change, round_conductor_internal_impedance,
    },
    coupled::fsi::{
        added_mass_piston_frequency, fluid_column_added_mass, moens_korteweg_wave_speed,
        oscillating_sphere_impedance,
    },
    coupled::mhd::{
        hartmann_velocity, hunt_wall_current_fraction, shercliff_layer_thickness,
        shercliff_side_layer_velocity, wall_conductance_ratio,
    },
    fluid::lubrication::{
        circular_squeeze_force, circular_squeeze_pressure, linear_slider_load,
        linear_slider_pressure, sommerfeld_journal_pressure,
    },
    fluid::phase_field_flow::{
        capillary_wave_frequency, laplace_pressure_jump, planar_interface_phase,
        planar_interface_surface_energy,
    },
    fluid::reacting_flow::{
        FrankKamenetskiiGeometry, burke_schumann_flame_location, burke_schumann_flame_sheet,
        fisher_kpp_minimum_wave_speed, frank_kamenetskii_critical_parameter,
    },
    thermal::radiation_transport::{
        chandrasekhar_h, homogeneous_slab_intensity, marshak_diffusion_energy,
    },
    thermal::thermoelectric::{
        compatibility_factor, constant_property_leg, maximum_generator_efficiency,
    },
};

#[test]
fn fsi_literature_cases_are_public_and_dimensionally_consistent() {
    // Causin-Gerbeau-Nobile added-mass piston.
    let added_mass = fluid_column_added_mass(1000.0, 0.01, 1.0);
    assert_eq!(added_mass, 10.0);
    let dry = added_mass_piston_frequency(1.0, 100.0, 1000.0, 1.0, 0.0);
    let wet = added_mass_piston_frequency(1.0, 100.0, 1000.0, 0.01, 1.0);
    assert!(wet < dry);

    // Moens-Korteweg pulse wave.
    let wave_speed = moens_korteweg_wave_speed(1.0e6, 1.0e-3, 1000.0, 0.01, 0.3);
    let stiffer = moens_korteweg_wave_speed(4.0e6, 1.0e-3, 1000.0, 0.01, 0.3);
    assert!((stiffer / wave_speed - 2.0).abs() < 1e-12);

    // Basset-Boussinesq-Oseen oscillating sphere.
    let steady = oscillating_sphere_impedance(0.01, 1000.0, 1e-3, 0.0);
    let harmonic = oscillating_sphere_impedance(0.01, 1000.0, 1e-3, 100.0);
    assert!((steady.re - 6.0 * PI * 1e-3 * 0.01).abs() < 1e-15);
    assert!(harmonic.re > steady.re && harmonic.im > 0.0);
}

#[test]
fn conjugate_heat_transfer_literature_cases_are_public() {
    // Luikov plane-wall conjugate solution.
    let wall = plane_wall_solution(400.0, 300.0, 100.0, 50.0, 0.02, 10.0);
    assert!((100.0 * (400.0 - wall.hot_interface_temperature) - wall.heat_flux).abs() < 1e-10);
    assert!((50.0 * (wall.cold_interface_temperature - 300.0) - wall.heat_flux).abs() < 1e-10);

    // Graetz fully developed limits and bulk-energy solution.
    assert_eq!(
        fully_developed_graetz_nusselt(GraetzBoundaryCondition::ConstantWallTemperature),
        3.66
    );
    assert!(
        (fully_developed_graetz_nusselt(GraetzBoundaryCondition::ConstantWallHeatFlux)
            - 48.0 / 11.0)
            .abs()
            < 1e-14
    );
    assert!((300.0..400.0).contains(&graetz_bulk_temperature(
        1.0, 300.0, 400.0, 10.0, 2.0, 1.0, 1000.0,
    )));

    // Gardner adiabatic-tip fin.
    let base = adiabatic_fin_temperature_excess(0.0, 0.1, 100.0, 20.0, 0.04, 200.0, 1e-4);
    assert!((base - 100.0).abs() < 1e-12);
    assert!(adiabatic_fin_base_heat_rate(0.1, 100.0, 20.0, 0.04, 200.0, 1e-4) > 0.0);
}

#[test]
fn electrothermal_literature_cases_are_public() {
    // Uniform DC-heated bar.
    let bar = dc_heated_bar(0.5, 1.0, 2.0, 300.0, 5.0, 10.0);
    assert_eq!(bar.potential, 1.0);
    assert_eq!(bar.temperature, 300.25);
    assert_eq!(bar.joule_heat_density, 20.0);

    // Dodd-Deeds coil above a conducting half-space.
    let mu0 = 4e-7 * PI;
    let air = dodd_deeds_filament_impedance_change(1e4, 0.01, 0.002, 0.0, mu0, 500);
    let conductor = dodd_deeds_filament_impedance_change(1e4, 0.01, 0.002, 5.8e7, mu0, 1000);
    assert_eq!(air, Complex64::ZERO);
    assert!(conductor.re > 0.0 && conductor.im < 0.0);

    // Bessel-function round conductor skin effect.
    let low = round_conductor_internal_impedance(1.0, 1e-3, 5.8e7, mu0);
    let high = round_conductor_internal_impedance(1e6, 1e-3, 5.8e7, mu0);
    assert!(high.re > low.re && high.im > 0.0);
}

#[test]
fn thermoelectric_literature_cases_are_public() {
    // Domenicali constant-property leg.
    let leg = constant_property_leg(0.01, 0.02, 500.0, 300.0, 2e5, 1e-5, 1.5, 200e-6);
    assert!(leg.temperature > 400.0);

    // Ioffe maximum efficiency.
    let eta_one = maximum_generator_efficiency(500.0, 300.0, 1.0);
    let eta_ten = maximum_generator_efficiency(500.0, 300.0, 10.0);
    assert!(eta_one > 0.0 && eta_ten > eta_one && eta_ten < 0.4);

    // Snyder-Ursell compatibility.
    let expected = (2.0_f64.sqrt() - 1.0) / (200e-6 * 400.0);
    assert!((compatibility_factor(200e-6, 400.0, 1.0) - expected).abs() < 1e-12);
}

#[test]
fn battery_literature_cases_are_public() {
    // Crank constant-flux spherical particle.
    let average = spherical_particle_average_concentration(30_000.0, 1e-5, 1e-3, 1.0);
    let surface = spherical_particle_surface_concentration(30_000.0, 1e-5, 1e-3, 1e-8, 1.0, 100);
    assert!(surface < average);

    // Sand depletion time.
    let base = sand_depletion_time(1e-9, 1000.0, 1.0);
    let doubled_flux = sand_depletion_time(1e-9, 1000.0, 2.0);
    assert!((base / doubled_flux - 4.0).abs() < 1e-12);

    // de Levie porous-electrode impedance.
    let impedance = de_levie_impedance(1e6, 10.0, 1e-2, 1.0);
    assert!(impedance.re > 0.0 && impedance.im < 0.0);
    assert!((impedance.re.abs() - impedance.im.abs()).abs() / impedance.re < 1e-6);
}

#[test]
fn mhd_literature_cases_are_public() {
    // Hartmann channel.
    let center = hartmann_velocity(0.0, 0.1, -100.0, 1e6, 0.5, 10.0);
    let wall = hartmann_velocity(0.1, 0.1, -100.0, 1e6, 0.5, 10.0);
    assert!(center > 0.0 && wall.abs() < 1e-12);

    // Shercliff side layer.
    assert_eq!(shercliff_layer_thickness(1.0, 100.0), 0.1);
    assert_eq!(shercliff_side_layer_velocity(0.0, 1.0, 100.0, 2.0), 0.0);

    // Hunt conducting-wall current closure.
    let conductance = wall_conductance_ratio(2.0, 0.1, 1.0, 0.2);
    assert_eq!(conductance, 1.0);
    assert_eq!(hunt_wall_current_fraction(conductance), 0.5);
}

#[test]
fn lubrication_literature_cases_are_public() {
    // Reynolds slider bearing.
    let pressure = linear_slider_pressure(0.05, 0.1, 100e-6, 50e-6, 0.08, 3.0);
    let load = linear_slider_load(0.1, 100e-6, 50e-6, 0.08, 3.0);
    assert!(pressure > 0.0 && load > 0.0);

    // Full-Sommerfeld journal bearing.
    let journal = sommerfeld_journal_pressure(0.7, 0.05, 50e-6, 0.4, 0.02, 2.0);
    let mirrored = sommerfeld_journal_pressure(-0.7, 0.05, 50e-6, 0.4, 0.02, 2.0);
    assert!((journal + mirrored).abs() < 1e-10 * journal.abs().max(1.0));

    // Stefan squeeze film.
    let squeeze_pressure = circular_squeeze_pressure(0.0, 0.02, 100e-6, -2e-4, 0.1);
    let squeeze_force = circular_squeeze_force(0.02, 100e-6, -2e-4, 0.1);
    assert!(squeeze_pressure > 0.0 && squeeze_force > 0.0);
}

#[test]
fn phase_field_literature_cases_are_public() {
    // Cahn-Hilliard planar interface and energy.
    assert_eq!(planar_interface_phase(0.0, 0.01), 0.0);
    let energy = planar_interface_surface_energy(0.07, 0.01, 12.0, 10_000);
    assert!((energy - 0.07).abs() / 0.07 < 1e-10);

    // Laplace pressure.
    assert_eq!(laplace_pressure_jump(0.1, 0.5, 2), 0.2);
    assert_eq!(laplace_pressure_jump(0.1, 0.5, 3), 0.4);

    // Capillary wave dispersion.
    let base = capillary_wave_frequency(0.07, 2.0, 1000.0, 1.0);
    let doubled = capillary_wave_frequency(0.07, 4.0, 1000.0, 1.0);
    assert!((doubled / base - 2.0_f64.powf(1.5)).abs() < 1e-12);
}

#[test]
fn reacting_flow_literature_cases_are_public() {
    // Fisher-KPP front speed.
    assert_eq!(fisher_kpp_minimum_wave_speed(2.0, 8.0), 8.0);

    // Burke-Schumann flame sheet.
    let location = burke_schumann_flame_location(1.0, 0.25);
    let flame = burke_schumann_flame_sheet(location, 1.0, 0.25, 300.0, 2200.0);
    assert_eq!(flame.fuel, 0.0);
    assert_eq!(flame.oxidizer, 0.0);
    assert_eq!(flame.temperature, 2200.0);

    // Frank-Kamenetskii critical parameters.
    let slab = frank_kamenetskii_critical_parameter(FrankKamenetskiiGeometry::PlaneSlab);
    let cylinder = frank_kamenetskii_critical_parameter(FrankKamenetskiiGeometry::InfiniteCylinder);
    let sphere = frank_kamenetskii_critical_parameter(FrankKamenetskiiGeometry::Sphere);
    assert_eq!((slab, cylinder, sphere), (0.878, 2.0, 3.32));
}

#[test]
fn radiation_transport_literature_cases_are_public() {
    // Formal homogeneous-slab solution.
    let slab = homogeneous_slab_intensity(0.2, 0.5, 3.0, 0.0);
    assert!((slab - 3.0 * (-0.4_f64).exp()).abs() < 1e-14);

    // Chandrasekhar isotropic half-space H function.
    assert_eq!(chandrasekhar_h(0.0, 1.0, 64, 100), 1.0);
    let endpoint = chandrasekhar_h(1.0, 1.0, 128, 3000);
    assert!((endpoint - 2.907_81).abs() < 4e-3);

    // Marshak diffusion-wave similarity solution.
    let first = marshak_diffusion_energy(1.0, 1.0, 1.0, 4.0);
    let similar = marshak_diffusion_energy(2.0, 4.0, 1.0, 4.0);
    assert!((first - similar).abs() < 1e-14);
}
