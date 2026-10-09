use sonium_sim_pcfan::PcFanConfig;
use std::path::PathBuf;

fn main() {
    let path = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/tmp/pcfan_geometry.vtu"));

    let config = PcFanConfig::default_120mm();
    let design = config.default_design_variables();
    let geometry = config.build_geometry(&design);

    geometry.write_vtu(&path).expect("write vtu");
    println!("wrote {}", path.display());
}
