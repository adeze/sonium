use serde::Deserialize;
use sonium_fem::physics::{
    coupled::battery, coupled::conjugate_heat_transfer as cht, coupled::electrothermal,
    coupled::fsi, coupled::mhd, fluid::lubrication, fluid::phase_field_flow, fluid::reacting_flow,
    thermal::radiation_transport, thermal::thermoelectric,
};
use sonium_qa::subprocess::WolframCommandExt;
use std::{collections::BTreeMap, env, path::PathBuf, process::Command, sync::OnceLock};

const MODEL_CASES: &[(&str, &[&str])] = &[
    (
        "fsi",
        &["added_mass_piston", "moens_korteweg", "oscillating_sphere"],
    ),
    (
        "conjugate_heat_transfer",
        &["plane_wall", "graetz", "adiabatic_fin"],
    ),
    (
        "electrothermal",
        &["dc_heated_bar", "round_conductor", "dodd_deeds"],
    ),
    (
        "thermoelectric",
        &["constant_leg", "ioffe_efficiency", "compatibility"],
    ),
    (
        "battery",
        &["spherical_particle", "sand_depletion", "de_levie"],
    ),
    ("mhd", &["hartmann", "shercliff", "hunt"]),
    (
        "lubrication",
        &["linear_slider", "sommerfeld", "squeeze_film"],
    ),
    (
        "phase_field_flow",
        &["planar_interface", "laplace_pressure", "capillary_wave"],
    ),
    (
        "reacting_flow",
        &["fisher_kpp", "burke_schumann", "frank_kamenetskii"],
    ),
    (
        "radiation_transport",
        &["absorbing_slab", "chandrasekhar_h", "marshak_wave"],
    ),
];

#[derive(Debug, Deserialize)]
struct ReferenceSet {
    schema_version: u32,
    cases: BTreeMap<String, Vec<f64>>,
}

static REFERENCES: OnceLock<Option<ReferenceSet>> = OnceLock::new();

fn script_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("wolfram/multiphysics.wls")
}

fn expected_case_ids() -> impl Iterator<Item = String> {
    MODEL_CASES
        .iter()
        .flat_map(|(model, cases)| cases.iter().map(move |case| format!("{model}.{case}")))
}

fn load_references() -> Option<ReferenceSet> {
    let Some(executable) = env::var_os("WOLFRAMSCRIPT") else {
        return Some(
            serde_json::from_str(include_str!("../wolfram/goldens/multiphysics.json"))
                .expect("parse checked-in Wolfram multiphysics reference"),
        );
    };
    let mut command = Command::new(&executable);
    command.arg("-file").arg(script_path());
    #[cfg(target_os = "macos")]
    if env::var_os("WolframKernel").is_none() {
        let bundled_kernel = PathBuf::from(
            "/Applications/Wolfram Engine.app/Contents/Resources/Wolfram Player.app/Contents/MacOS/WolframKernel",
        );
        if bundled_kernel.is_file() {
            command.env("WolframKernel", bundled_kernel);
        }
    }
    let output = match command.wolfram_output() {
        Ok(output) => output,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            eprintln!(
                "Wolfram Engine not available; skipping optional multiphysics differential tests"
            );
            return None;
        }
        Err(error) => panic!("failed to launch {executable:?}: {error}"),
    };
    assert!(
        output.status.success(),
        "Wolfram reference script failed:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    let json = stdout
        .lines()
        .rev()
        .find(|line| line.trim_start().starts_with('{'))
        .unwrap_or(&stdout);
    let references: ReferenceSet = serde_json::from_str(json).unwrap_or_else(|error| {
        panic!(
            "Wolfram output was not valid reference JSON: {error}\nstdout:\n{}\nstderr:\n{}",
            stdout,
            String::from_utf8_lossy(&output.stderr)
        )
    });
    assert_eq!(
        references.schema_version, 1,
        "unsupported Wolfram reference schema"
    );
    let expected = expected_case_ids().collect::<Vec<_>>();
    assert_eq!(
        references.cases.len(),
        expected.len(),
        "unexpected Wolfram reference case count"
    );
    for id in expected {
        assert!(
            references.cases.contains_key(&id),
            "Wolfram output omitted {id}"
        );
    }
    Some(references)
}

fn compare(case: &str, actual: &[f64], relative_tolerance: f64, absolute_tolerance: f64) {
    let Some(references) = REFERENCES.get_or_init(load_references) else {
        return;
    };
    let expected = references
        .cases
        .get(case)
        .unwrap_or_else(|| panic!("missing Wolfram case {case}"));
    assert_eq!(
        actual.len(),
        expected.len(),
        "component count differs for {case}"
    );
    for (component, (&got, &want)) in actual.iter().zip(expected).enumerate() {
        let tolerance = absolute_tolerance + relative_tolerance * got.abs().max(want.abs());
        assert!(
            (got - want).abs() <= tolerance,
            "{case} component {component}: Sonium={got:.16e}, Wolfram={want:.16e}, tolerance={tolerance:.3e}"
        );
    }
}

#[test]
fn wolfram_manifest_has_three_cases_per_physics_model() {
    let script = std::fs::read_to_string(script_path()).expect("read Wolfram reference script");
    assert_eq!(MODEL_CASES.len(), 10);
    for (model, cases) in MODEL_CASES {
        assert_eq!(
            cases.len(),
            3,
            "{model} must have exactly three Wolfram cases"
        );
        for case in *cases {
            let id = format!("\"{model}.{case}\" ->");
            assert_eq!(
                script.matches(&id).count(),
                1,
                "Wolfram script must declare {model}.{case} once"
            );
        }
    }
}

#[test]
fn wolfram_fsi_added_mass_piston() {
    compare(
        "fsi.added_mass_piston",
        &[
            fsi::fluid_column_added_mass(1000.0, 0.01, 1.0),
            fsi::added_mass_piston_frequency(1.0, 100.0, 1000.0, 0.01, 1.0),
        ],
        1e-13,
        1e-13,
    );
}

#[test]
fn wolfram_fsi_moens_korteweg() {
    compare(
        "fsi.moens_korteweg",
        &[fsi::moens_korteweg_wave_speed(1e6, 1e-3, 1000.0, 0.01, 0.3)],
        1e-13,
        1e-13,
    );
}

#[test]
fn wolfram_fsi_oscillating_sphere() {
    let value = fsi::oscillating_sphere_impedance(0.01, 1000.0, 1e-3, 100.0);
    compare(
        "fsi.oscillating_sphere",
        &[value.re, value.im],
        1e-12,
        1e-14,
    );
}

#[test]
fn wolfram_cht_plane_wall() {
    let value = cht::plane_wall_solution(400.0, 300.0, 100.0, 50.0, 0.02, 10.0);
    compare(
        "conjugate_heat_transfer.plane_wall",
        &[
            value.heat_flux,
            value.hot_interface_temperature,
            value.cold_interface_temperature,
            value.overall_heat_transfer_coefficient,
        ],
        1e-13,
        1e-12,
    );
}

#[test]
fn wolfram_cht_graetz() {
    compare(
        "conjugate_heat_transfer.graetz",
        &[
            cht::fully_developed_graetz_nusselt(
                cht::GraetzBoundaryCondition::ConstantWallTemperature,
            ),
            cht::fully_developed_graetz_nusselt(cht::GraetzBoundaryCondition::ConstantWallHeatFlux),
            cht::graetz_bulk_temperature(1.0, 300.0, 400.0, 10.0, 2.0, 1.0, 1000.0),
        ],
        1e-13,
        1e-12,
    );
}

#[test]
fn wolfram_cht_adiabatic_fin() {
    compare(
        "conjugate_heat_transfer.adiabatic_fin",
        &[
            cht::adiabatic_fin_temperature_excess(0.04, 0.1, 100.0, 20.0, 0.04, 200.0, 1e-4),
            cht::adiabatic_fin_base_heat_rate(0.1, 100.0, 20.0, 0.04, 200.0, 1e-4),
        ],
        1e-12,
        1e-12,
    );
}

#[test]
fn wolfram_electrothermal_dc_heated_bar() {
    let value = electrothermal::dc_heated_bar(0.5, 1.0, 2.0, 300.0, 5.0, 10.0);
    compare(
        "electrothermal.dc_heated_bar",
        &[value.potential, value.temperature, value.joule_heat_density],
        1e-13,
        1e-13,
    );
}

#[test]
fn wolfram_electrothermal_round_conductor() {
    let value = electrothermal::round_conductor_internal_impedance(
        1e6,
        1e-3,
        5.8e7,
        4e-7 * std::f64::consts::PI,
    );
    compare(
        "electrothermal.round_conductor",
        &[value.re, value.im],
        2e-11,
        1e-13,
    );
}

#[test]
fn wolfram_electrothermal_dodd_deeds() {
    let value = electrothermal::dodd_deeds_filament_impedance_change(
        1e4,
        5e-4,
        5e-3,
        5.8e7,
        4e-7 * std::f64::consts::PI,
        20_000,
    );
    compare(
        "electrothermal.dodd_deeds",
        &[value.re, value.im],
        2e-5,
        1e-12,
    );
}

#[test]
fn wolfram_thermoelectric_constant_leg() {
    let value =
        thermoelectric::constant_property_leg(0.01, 0.02, 500.0, 300.0, 2e5, 1e-5, 1.5, 200e-6);
    compare(
        "thermoelectric.constant_leg",
        &[value.temperature, value.heat_flux, value.potential],
        1e-12,
        1e-11,
    );
}

#[test]
fn wolfram_thermoelectric_ioffe_efficiency() {
    compare(
        "thermoelectric.ioffe_efficiency",
        &[thermoelectric::maximum_generator_efficiency(
            500.0, 300.0, 1.0,
        )],
        1e-13,
        1e-14,
    );
}

#[test]
fn wolfram_thermoelectric_compatibility() {
    compare(
        "thermoelectric.compatibility",
        &[thermoelectric::compatibility_factor(200e-6, 400.0, 1.0)],
        1e-13,
        1e-13,
    );
}

#[test]
fn wolfram_battery_spherical_particle() {
    compare(
        "battery.spherical_particle",
        &[
            battery::spherical_particle_surface_concentration(30_000.0, 1e-5, 1e-3, 1e-8, 1.0, 100),
            battery::spherical_particle_average_concentration(30_000.0, 1e-5, 1e-3, 1.0),
        ],
        2e-12,
        1e-9,
    );
}

#[test]
fn wolfram_battery_sand_depletion() {
    compare(
        "battery.sand_depletion",
        &[battery::sand_depletion_time(1e-9, 1000.0, 1.0)],
        1e-13,
        1e-14,
    );
}

#[test]
fn wolfram_battery_de_levie() {
    let value = battery::de_levie_impedance(1e6, 10.0, 1e-2, 1.0);
    compare("battery.de_levie", &[value.re, value.im], 1e-12, 1e-14);
}

#[test]
fn wolfram_mhd_hartmann() {
    compare(
        "mhd.hartmann",
        &[
            mhd::hartmann_velocity(0.0, 0.1, -100.0, 1e6, 0.5, 10.0),
            mhd::hartmann_mean_velocity_factor(10.0),
        ],
        1e-12,
        1e-14,
    );
}

#[test]
fn wolfram_mhd_shercliff() {
    compare(
        "mhd.shercliff",
        &[
            mhd::shercliff_layer_thickness(1.0, 100.0),
            mhd::shercliff_side_layer_velocity(0.05, 1.0, 100.0, 2.0),
        ],
        1e-13,
        1e-14,
    );
}

#[test]
fn wolfram_mhd_hunt() {
    let ratio = mhd::wall_conductance_ratio(2.0, 0.1, 1.0, 0.2);
    compare(
        "mhd.hunt",
        &[ratio, mhd::hunt_wall_current_fraction(ratio)],
        1e-13,
        1e-14,
    );
}

#[test]
fn wolfram_lubrication_linear_slider() {
    compare(
        "lubrication.linear_slider",
        &[
            lubrication::linear_slider_pressure(0.05, 0.1, 100e-6, 50e-6, 0.08, 3.0),
            lubrication::linear_slider_load(0.1, 100e-6, 50e-6, 0.08, 3.0),
        ],
        2e-12,
        1e-8,
    );
}

#[test]
fn wolfram_lubrication_sommerfeld() {
    compare(
        "lubrication.sommerfeld",
        &[lubrication::sommerfeld_journal_pressure(
            0.7, 0.05, 50e-6, 0.4, 0.02, 2.0,
        )],
        1e-12,
        1e-8,
    );
}

#[test]
fn wolfram_lubrication_squeeze_film() {
    compare(
        "lubrication.squeeze_film",
        &[
            lubrication::circular_squeeze_pressure(0.01, 0.02, 100e-6, -2e-4, 0.1),
            lubrication::circular_squeeze_force(0.02, 100e-6, -2e-4, 0.1),
        ],
        1e-12,
        1e-9,
    );
}

#[test]
fn wolfram_phase_field_planar_interface() {
    compare(
        "phase_field_flow.planar_interface",
        &[
            phase_field_flow::planar_interface_phase(0.1, 0.01),
            phase_field_flow::planar_interface_surface_energy(0.07, 0.01, 12.0, 20_000),
        ],
        2e-10,
        1e-12,
    );
}

#[test]
fn wolfram_phase_field_laplace_pressure() {
    compare(
        "phase_field_flow.laplace_pressure",
        &[
            phase_field_flow::laplace_pressure_jump(0.1, 0.5, 2),
            phase_field_flow::laplace_pressure_jump(0.1, 0.5, 3),
        ],
        1e-13,
        1e-14,
    );
}

#[test]
fn wolfram_phase_field_capillary_wave() {
    compare(
        "phase_field_flow.capillary_wave",
        &[phase_field_flow::capillary_wave_frequency(
            0.07, 2.0, 1000.0, 1.0,
        )],
        1e-13,
        1e-14,
    );
}

#[test]
fn wolfram_reacting_fisher_kpp() {
    compare(
        "reacting_flow.fisher_kpp",
        &[reacting_flow::fisher_kpp_minimum_wave_speed(2.0, 8.0)],
        1e-13,
        1e-14,
    );
}

#[test]
fn wolfram_reacting_burke_schumann() {
    let location = reacting_flow::burke_schumann_flame_location(1.0, 0.25);
    let value = reacting_flow::burke_schumann_flame_sheet(location, 1.0, 0.25, 300.0, 2200.0);
    compare(
        "reacting_flow.burke_schumann",
        &[
            value.mixture_fraction,
            value.fuel,
            value.oxidizer,
            value.temperature,
            location,
        ],
        1e-13,
        1e-13,
    );
}

#[test]
fn wolfram_reacting_frank_kamenetskii() {
    compare(
        "reacting_flow.frank_kamenetskii",
        &[
            reacting_flow::frank_kamenetskii_critical_parameter(
                reacting_flow::FrankKamenetskiiGeometry::PlaneSlab,
            ),
            reacting_flow::frank_kamenetskii_critical_parameter(
                reacting_flow::FrankKamenetskiiGeometry::InfiniteCylinder,
            ),
            reacting_flow::frank_kamenetskii_critical_parameter(
                reacting_flow::FrankKamenetskiiGeometry::Sphere,
            ),
        ],
        1e-13,
        1e-14,
    );
}

#[test]
fn wolfram_radiation_absorbing_slab() {
    compare(
        "radiation_transport.absorbing_slab",
        &[radiation_transport::homogeneous_slab_intensity(
            0.2, 0.5, 3.0, 0.0,
        )],
        1e-13,
        1e-14,
    );
}

#[test]
fn wolfram_radiation_chandrasekhar_h() {
    compare(
        "radiation_transport.chandrasekhar_h",
        &[radiation_transport::chandrasekhar_h(1.0, 1.0, 256, 5000)],
        7e-4,
        1e-5,
    );
}

#[test]
fn wolfram_radiation_marshak_wave() {
    compare(
        "radiation_transport.marshak_wave",
        &[radiation_transport::marshak_diffusion_energy(
            1.0, 1.0, 1.0, 4.0,
        )],
        4e-7,
        1e-7,
    );
}
