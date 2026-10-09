//! Golden `.geo` snapshot tests for high-risk string emission.
//!
//! Cyclic (`Rotate`/`Duplicata` loops), periodic, embed, and compound
//! controls are generated as raw Gmsh script text. Substring assertions
//! cannot catch reordered statements, duplicated blocks, renamed entities,
//! or altered numeric formatting, so these tests compare the entire `.geo`
//! output byte-for-byte against checked-in goldens.
//!
//! The goldens pin the compiler version banner too: bump the crate version
//! and regenerate them deliberately (via the `meshlang` CLI) rather than
//! editing by hand. Each golden was verified by inspection when checked in:
//! sector loop `1:11` plus original = 12 sectors, single `Periodic Surface`
//! with the translation matrix, exact `In Volume` / `Compound Volume` forms.

fn assert_golden(source: &str, filename: &str, golden: &str) {
    let geo =
        meshlang_compiler::compile_to_geo(source, filename).expect("golden source must compile");
    assert_eq!(
        geo, golden,
        "golden .geo mismatch for {filename}: regenerate deliberately, do not hand-edit"
    );
}

#[test]
fn cyclic_symmetry_geo_matches_golden() {
    let source = "geometry test { let rotor = Box(1.0, 1.0, 0.2) }\n\
        design_region rotor_design {\n\
            kind: design\n\
            region: test.rotor\n\
            annotations: [symmetry(kind: cyclic(axis: [0.0, 0.0, 1.0], sectors: 12))]\n\
        }\n\
        export { format: msh4 filename: \"test.msh\" }\n";
    assert_golden(
        source,
        "cyclic-backend.mlg",
        include_str!("golden/cyclic.geo"),
    );
}

#[test]
fn periodic_geo_matches_golden() {
    let source = "geometry model {\n\
            let source = Rectangle(0.5, 0.5)\n\
            let target = Rectangle(0.5, 0.5) |> translate(1.0, 0.0, 0.0)\n\
        }\n\
        mesh model {\n\
            periodic surface(model.source, model.target, transform: [1.0, 0.0, 0.0])\n\
        }\n\
        export { format: msh4 filename: \"periodic.msh\" }\n";
    assert_golden(source, "periodic.mlg", include_str!("golden/periodic.geo"));
}

#[test]
fn embed_geo_matches_golden() {
    let source = "geometry model {\n\
            let body = Box(1.0, 1.0, 1.0)\n\
            let face = Rectangle(0.5, 0.5)\n\
        }\n\
        mesh model {\n\
            embed surface(model.face) in volume(model.body)\n\
        }\n\
        export { format: msh4 filename: \"embed.msh\" }\n";
    assert_golden(source, "embed.mlg", include_str!("golden/embed.geo"));
}

#[test]
fn compound_geo_matches_golden() {
    let source = "geometry pieces {\n\
            let left = Box(1.0, 1.0, 1.0)\n\
            let right = Box(1.0, 1.0, 1.0) |> translate(1.0, 0.0, 0.0)\n\
        }\n\
        mesh for pieces {\n\
            compound volume(pieces.left, pieces.right)\n\
        }\n\
        export { format: msh4  filename: \"compound.msh\" }\n";
    assert_golden(source, "compound.mlg", include_str!("golden/compound.geo"));
}
