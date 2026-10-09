//! Axisymmetric shrouded propeller comparison with toroidal and serrated source models.
//!
//! Run with:
//! `cargo run -p sonium-sim-propellers --release --example axisymmetric_shrouded_toroidal`

use sonium_fem::optimization::shroud_opt::{
    ShroudOptConfig, ShroudProfile, evaluate_composite_objective,
};
use sonium_fem::physics::acoustics::aeroacoustics::PropellerSource;
use sonium_sim_propellers::{
    EdgeTreatment, EvaluationMode, PropellerDesign, PropellerPlanform, SerrationGeometry,
};

fn main() {
    let source = PropellerSource::drone_5inch(18_000.0);
    let n_harmonics = 20;

    println!("=== Shrouded Propeller Noise Design Study ===\n");
    println!(
        "Source: {} blades, {:.0} RPM, BPF {:.0} Hz, tip Mach {:.3}",
        source.n_blades,
        source.rpm,
        source.blade_passing_frequency(),
        source.tip_mach()
    );
    println!("Axisymmetric propagation: shroud Helmholtz model");
    println!("3D blade realism: source-side toroidal/serration modifiers\n");

    let shroud = ShroudProfile::straight(0.070, 0.0, 0.15, 5);
    let config = ShroudOptConfig {
        propeller: source.clone(),
        n_harmonics: 6,
        nz: 48,
        nr: 10,
        ..Default::default()
    };
    let (_, shroud_swl, drag, curv, vol) = evaluate_composite_objective(&shroud, &config);

    println!("Straight shroud propagation control:");
    println!("  SWL_A = {shroud_swl:.1} dB");
    println!("  drag penalty = {drag:.4}");
    println!("  curvature penalty = {curv:.4}");
    println!("  volume = {:.2} cm3\n", vol * 1e6);

    let designs = [
        (
            "conventional",
            PropellerDesign::conventional(source.clone()),
        ),
        (
            "toroidal loop",
            PropellerDesign {
                source: source.clone(),
                planform: PropellerPlanform::ToroidalLoop,
                edge_treatment: EdgeTreatment::Smooth,
                serrations: SerrationGeometry::default(),
            },
        ),
        (
            "LE+TE serrated",
            PropellerDesign {
                source: source.clone(),
                planform: PropellerPlanform::Conventional,
                edge_treatment: EdgeTreatment::LeadingAndTrailingSerrations,
                serrations: SerrationGeometry::owl_like_micro_airfoil(),
            },
        ),
        (
            "toroidal + serrated",
            PropellerDesign::toroidal_serrated(source.clone()),
        ),
    ];

    let baseline_fast = designs[0]
        .1
        .spectrum(n_harmonics, EvaluationMode::FastSurrogate);
    let baseline_research = designs[0]
        .1
        .spectrum(n_harmonics, EvaluationMode::ResearchControl);

    println!(
        "{:<22} {:>12} {:>14} {:>14}",
        "design", "mode", "1-5 kHz reduction", "min amp ratio"
    );
    println!("{:-<68}", "");
    for (name, design) in &designs {
        for mode in [
            EvaluationMode::FastSurrogate,
            EvaluationMode::ResearchControl,
        ] {
            let spectrum = design.spectrum(n_harmonics, mode);
            let baseline = match mode {
                EvaluationMode::FastSurrogate => &baseline_fast,
                EvaluationMode::ResearchControl => &baseline_research,
            };
            let reduction = spectrum.band_reduction_db_against(baseline, 1_000.0, 5_000.0);
            let min_ratio = spectrum
                .harmonics
                .iter()
                .map(|h| h.amplitude_reduction)
                .fold(1.0_f64, f64::min);
            println!(
                "{:<22} {:>12?} {:>+14.2} {:>14.3}",
                name, mode, reduction, min_ratio
            );
        }
    }

    println!("\nCombined research-control harmonics:");
    println!(
        "{:>4} {:>10} {:>14} {:>14} {:>10}",
        "h", "freq(Hz)", "base peak Pa", "design peak Pa", "ratio"
    );
    let combined = designs[3]
        .1
        .spectrum(n_harmonics, EvaluationMode::ResearchControl);
    for harmonic in combined.harmonics {
        println!(
            "{:>4} {:>10.0} {:>14.4e} {:>14.4e} {:>10.3}",
            harmonic.harmonic,
            harmonic.frequency_hz,
            harmonic.base_pressure_pa,
            harmonic.pressure_pa,
            harmonic.amplitude_reduction
        );
    }
}
