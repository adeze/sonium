use sonium_sim_pcfan::meshlang_geometry::MeshlangPcfanGeometry;

fn main() {
    let geom = MeshlangPcfanGeometry::compile().expect("compile");
    print!("{}", geom.code);
    eprintln!("--- diagnostics: {} ---", geom.diagnostics.len());
    for d in &geom.diagnostics {
        eprintln!("{:?}", d);
    }
}
