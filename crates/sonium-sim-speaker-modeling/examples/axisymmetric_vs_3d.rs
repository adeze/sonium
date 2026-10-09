use sonium_sim_speaker_modeling::{
    AcousticMedium, SpeakerDriverStudy, StudyRequirements, choose_symmetry, octave_sweep,
};

fn main() {
    let base = SpeakerDriverStudy::detailed_frequency_domain_driver();

    let mut sector_geometry = base.geometry.clone();
    sector_geometry.sector_angle_degrees = Some(30.0);

    let mut full_geometry = base.geometry.clone();
    full_geometry.has_non_axisymmetric_basket = true;

    let studies = [
        ("axisymmetric", base.geometry.clone()),
        ("sector-modes", sector_geometry),
        ("full-3d-basket", full_geometry),
    ];

    for (name, geometry) in studies {
        let requirements = StudyRequirements {
            name: name.to_string(),
            geometry,
            voice_coil: base.voice_coil,
            moving_system: base.moving_system,
            harmonic_small_signal: true,
            transient_or_pulse: false,
            nonlinear_large_signal: false,
            frequency_hz: octave_sweep(100.0, 2_000.0, 3),
            voltage_rms_v: 2.83,
            acoustic_medium: AcousticMedium::standard_air(),
        };

        println!("{name}: {:?}", choose_symmetry(&requirements.geometry));
    }
}
