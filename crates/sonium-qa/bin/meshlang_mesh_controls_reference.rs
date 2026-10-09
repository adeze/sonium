//! CPU-only canonical mesh-control parity reference.

use meshlang_compiler::{compile_to_geo_full, compile_to_python_full};
use sonium_qa::QaResult;
use std::time::Instant;

const CASE_ID: &str = "meshlang.mesh-controls-reference.v1";
const SOURCE: &str = r#"
geometry model {
    let body = Box(1.0, 1.0, 1.0)
    let body2 = Box(1.0, 1.0, 1.0) |> translate(1.0, 0.0, 0.0)
    let face = Rectangle(0.5, 0.5)
    let face2 = Rectangle(0.5, 0.5) |> translate(1.0, 0.0, 0.0)
}
mesh model {
    transfinite surface(model.face, arrangement: "Left", corners: [1, 2, 3, 4])
    embed surface(model.face) in volume(model.body)
    periodic surface(model.face, model.face2, transform: [1.0, 0.0, 0.0])
    compound volume(model.body, model.body2)
    refine near(model.face, distance: 0.2) {
        size: 0.01
        growth_rate: 2.0
    }
    refine on(model.face) {
        size: 0.02
        order: 2
    }
    refine in(Box(0.2, 0.2, 0.2) |> translate(1.0, 2.0, 3.0)) {
        size: 0.03
        size_out: 0.1
    }
    refine field(distance_from: model.face) {
        size: 0.01
        size_max: 0.1
        distance_min: 0.2
        distance_max: 0.5
    }
    metric model.face {
        kind: isotropic
        size: 0.1
    }
    metric model.body {
        kind: anisotropic
        components: [1, 2, 3, 4, 5, 6]
    }
    boundary_layer on([model.body, model.body2]) {
        first_height: 0.001
        thickness: 0.004
        layers: 4
        growth_rate: 1.25
    }
}
"#;

fn run() -> Result<QaResult, String> {
    let started = Instant::now();
    let geo = compile_to_geo_full(SOURCE, "mesh-controls-reference.mlg")
        .map_err(|error| format!("GEO mesh-control compilation failed: {error}"))?;
    let python = compile_to_python_full(SOURCE, "mesh-controls-reference.mlg")
        .map_err(|error| format!("Python mesh-control compilation failed: {error}"))?;
    let canonical: serde_json::Value = serde_json::from_str(&geo.canonical_ir_json)
        .map_err(|error| format!("canonical IR JSON is invalid: {error}"))?;
    let mesh = canonical
        .pointer("/ir/mesh")
        .ok_or_else(|| "canonical IR has no mesh controls".to_owned())?;
    let count = |name: &str| {
        mesh.get(name)
            .and_then(serde_json::Value::as_array)
            .map_or(0, Vec::len)
    };
    let transfinite_count = count("transfinite");
    let embed_count = count("embeds");
    let compound_count = count("compounds");
    let boundary_layer_count = count("boundary_layers");
    let periodic_count = canonical
        .pointer("/ir/periodic_pairs")
        .and_then(serde_json::Value::as_array)
        .map_or(0, Vec::len);
    let refinement_count = count("refinements");
    let metric_count = count("metrics");
    if (
        transfinite_count,
        embed_count,
        periodic_count,
        compound_count,
        boundary_layer_count,
        refinement_count,
        metric_count,
    ) != (1, 1, 1, 1, 1, 4, 2)
    {
        return Err(format!(
            "canonical mesh-control counts drifted: transfinite={transfinite_count}, embed={embed_count}, periodic={periodic_count}, compound={compound_count}, boundary_layer={boundary_layer_count}, refine={refinement_count}, metric={metric_count}"
        ));
    }
    let geo_counts = [
        geo.code.matches("Transfinite Surface").count(),
        geo.code.matches(" In Volume").count(),
        geo.code.matches("Periodic Surface").count(),
        geo.code.matches("Compound Volume").count(),
        geo.code.matches("= BoundaryLayer;").count(),
        geo.code.matches("= Distance;").count(),
        geo.code.matches("= Threshold;").count(),
        geo.code.matches("= Box;").count(),
        geo.code.matches("= MathEval;").count(),
        geo.code.matches("= MathEvalAniso;").count(),
        geo.code.matches("= Restrict;").count(),
    ];
    let python_counts = [
        python
            .code
            .matches("gmsh.model.mesh.setTransfiniteSurface")
            .count(),
        python.code.matches("gmsh.model.mesh.embed(").count(),
        python.code.matches("gmsh.model.mesh.setPeriodic(").count(),
        python.code.matches("gmsh.model.mesh.setCompound(").count(),
        python.code.matches("field.add('BoundaryLayer')").count(),
        python.code.matches("field.add('Distance')").count(),
        python.code.matches("field.add('Threshold')").count(),
        python.code.matches("field.add('Box')").count(),
        python.code.matches("field.add('MathEval')").count(),
        python.code.matches("field.add('MathEvalAniso')").count(),
        python.code.matches("field.add('Restrict')").count(),
    ];
    if geo_counts != [1, 1, 1, 1, 1, 2, 2, 1, 1, 1, 2]
        || python_counts != [1, 1, 1, 1, 1, 2, 2, 1, 1, 1, 2]
    {
        return Err(format!(
            "canonical mesh-control emission counts drifted: geo={geo_counts:?}, python={python_counts:?}"
        ));
    }
    if !geo.code.contains(".Thickness = 0.004;") || !python.code.contains("'Thickness', 0.004") {
        return Err("canonical boundary-layer thickness emission drifted".into());
    }
    Ok(QaResult::passed(CASE_ID)
        .requested_backend("cpu")
        .actual_backend("cpu")
        .metric("canonical_transfinite", transfinite_count as f64)
        .metric("canonical_embeds", embed_count as f64)
        .metric("canonical_periodic", periodic_count as f64)
        .metric("canonical_compounds", compound_count as f64)
        .metric("canonical_boundary_layers", boundary_layer_count as f64)
        .metric("canonical_refinements", refinement_count as f64)
        .metric("canonical_metrics", metric_count as f64)
        .metric("geo_controls", geo_counts.iter().sum::<usize>() as f64)
        .metric(
            "python_controls",
            python_counts.iter().sum::<usize>() as f64,
        )
        .dimension("2-D/3-D mesh controls")
        .formulation("canonical mesh-control authority")
        .input_checksum("source:meshlang-mesh-controls-reference:v1")
        .reference_checksum("geo-python-canonical-mesh-controls-parity:v1")
        .seed("meshlang-mesh-controls-reference:cpu:v1")
        .elapsed_ms(started.elapsed().as_millis() as u64))
}

fn main() {
    let started = Instant::now();
    let (result, passed) = match run() {
        Ok(result) => (
            result.elapsed_ms(started.elapsed().as_millis() as u64),
            true,
        ),
        Err(error) => (
            QaResult::failed(CASE_ID, error).elapsed_ms(started.elapsed().as_millis() as u64),
            false,
        ),
    };
    println!(
        "{}",
        result.to_json_line().expect("serialize QA result JSON")
    );
    if !passed {
        println!("FAILED");
        std::process::exit(1);
    }
    println!("SUCCESS");
}
