use sonium_fem::mesh::{CCoreMeshConfig, CCoreRegion, c_core_mesh_triangles};

fn main() {
    let result = std::panic::catch_unwind(|| {
        let mesh = c_core_mesh_triangles(CCoreMeshConfig {
            width: 1.0,
            height: 0.8,
            core_thickness: 0.18,
            gap_height: 0.12,
            nx: 12,
            ny: 10,
        });
        assert!(mesh.element_regions.contains(&CCoreRegion::Coil));
        assert!(mesh.element_regions.contains(&CCoreRegion::Gap));
        println!(
            "Magnetostatics air gap C-core: elems = {}",
            mesh.mesh.num_elements()
        );
    });
    if result.is_ok() {
        println!("SUCCESS");
    } else {
        println!("FAILED");
        std::process::exit(1);
    }
}
