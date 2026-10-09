//! 3D toroidal propeller geometry and panel-source comparison.
//!
//! Run with:
//! `cargo run -p sonium-sim-propellers --release --example three_dimensional_toroidal`

use sonium_fem::physics::acoustics::aeroacoustics::PropellerSource;
use sonium_sim_propellers::{
    Blade3dConfig, EdgeTreatment, EvaluationMode, PropellerDesign, PropellerPlanform,
    SerrationGeometry, compare_panel_acoustics,
};

fn main() {
    let source = PropellerSource::drone_5inch(18_000.0);
    let baseline = PropellerDesign::conventional(source.clone());
    let toroidal = PropellerDesign {
        source: source.clone(),
        planform: PropellerPlanform::ToroidalLoop,
        edge_treatment: EdgeTreatment::Smooth,
        serrations: SerrationGeometry::default(),
    };
    let serrated = PropellerDesign {
        source: source.clone(),
        planform: PropellerPlanform::Conventional,
        edge_treatment: EdgeTreatment::LeadingAndTrailingSerrations,
        serrations: SerrationGeometry::owl_like_micro_airfoil(),
    };
    let combined = PropellerDesign::toroidal_serrated(source);

    let cfg = Blade3dConfig::research_control_64c();
    let baseline_mesh = cfg.build_mesh(&baseline);
    let baseline_spectrum = baseline.spectrum(20, EvaluationMode::ResearchControl);

    println!("=== 3D Toroidal Propeller Panel Study ===\n");
    println!(
        "mesh resolution: radial={}, chord={}, loop={} panels/blade",
        cfg.radial_segments, cfg.chord_segments, cfg.loop_segments
    );
    println!(
        "baseline panels: {}, area {:.5} m2\n",
        baseline_mesh.panels.len(),
        baseline_mesh.total_area()
    );
    println!(
        "{:<22} {:>10} {:>12} {:>12} {:>12} {:>12}",
        "design", "panels", "area m2", "tip area", "vortex", "reduction dB"
    );
    println!("{:-<86}", "");

    for (name, design) in [
        ("toroidal loop", toroidal),
        ("LE+TE serrated", serrated),
        ("toroidal + serrated", combined),
    ] {
        let mesh = cfg.build_mesh(&design);
        let spectrum = design.spectrum(20, EvaluationMode::ResearchControl);
        let summary = compare_panel_acoustics(&baseline_mesh, &mesh, &baseline_spectrum, &spectrum);

        println!(
            "{:<22} {:>10} {:>12.5} {:>12.3} {:>12.3} {:>+12.2}",
            name,
            mesh.panels.len(),
            summary.total_area,
            summary.tip_area_fraction,
            summary.vortex_source_index,
            summary.sensitive_band_reduction_db
        );
    }
}
