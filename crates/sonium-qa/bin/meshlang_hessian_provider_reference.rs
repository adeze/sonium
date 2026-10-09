//! Release-facing reference for the shared MeshLang Hessian metric provider.

use meshlang_ast::Expr;
use meshlang_compiler::{
    CompileOptions, HessianMetricProvider, HessianMetricRequest, compile_to_geo_full_with_options,
    compile_to_python_full_with_options,
};
use sonium_qa::QaResult;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

const CASE_ID: &str = "meshlang.hessian-provider-reference.v1";

#[derive(Debug, Default)]
struct ReferenceHessianProvider {
    requests: AtomicUsize,
}

impl HessianMetricProvider for ReferenceHessianProvider {
    fn resolve(&self, request: HessianMetricRequest<'_>) -> Result<[Expr; 6], String> {
        // A production provider can bridge sampled solver fields. This
        // deterministic fixture echoes an analytic scalar field into all six
        // tensor slots so both emitters can be compared independently.
        self.requests.fetch_add(1, Ordering::SeqCst);
        Ok(std::array::from_fn(|_| request.field.clone()))
    }
}

fn run() -> Result<QaResult, String> {
    let source = r#"
geometry model {
    let body = Box(1.0, 1.0, 1.0)
}
mesh for model {
    metric model.body {
        kind: hessian
        field: 1.0
        minimum_size: 0.01
        maximum_size: 0.1
    }
}
"#;
    let provider = Arc::new(ReferenceHessianProvider::default());
    let options = CompileOptions {
        hessian_metric_provider: Some(provider.clone()),
        ..CompileOptions::default()
    };
    let geo = compile_to_geo_full_with_options(source, "hessian-reference.mlg", &options)
        .map_err(|error| format!("GEO compile: {error}"))?;
    let python = compile_to_python_full_with_options(source, "hessian-reference.mlg", &options)
        .map_err(|error| format!("Python compile: {error}"))?;

    let component_names = ["M11", "M22", "M33", "M12", "M13", "M23"];
    let checks = [
        provider.requests.load(Ordering::SeqCst) == 2,
        geo.code.contains("MathEvalAniso"),
        python.code.contains("MathEvalAniso"),
        component_names.iter().all(|name| geo.code.contains(name)),
        component_names
            .iter()
            .all(|name| python.code.contains(name)),
        geo.canonical_ir_json == python.canonical_ir_json,
        geo.canonical_ir_hash == python.canonical_ir_hash,
        geo.sensitivity_manifest_json == python.sensitivity_manifest_json,
        geo.provenance_manifest_json == python.provenance_manifest_json,
    ];
    let checks_passed = checks.iter().filter(|passed| **passed).count();
    if checks_passed != checks.len() {
        return Err(format!(
            "Hessian provider/backend parity checks: {checks_passed}/{}",
            checks.len()
        ));
    }

    Ok(QaResult::passed(CASE_ID)
        .requested_backend("cpu")
        .actual_backend("cpu")
        .metric("checks_passed", checks_passed as f64)
        .metric(
            "provider_request_count",
            provider.requests.load(Ordering::SeqCst) as f64,
        )
        .minimum("checks_passed", checks.len() as f64)
        .minimum("provider_request_count", 2.0)
        .input_checksum("source:analytic-hessian-metric:meshlang-hessian-provider-v1")
        .reference("shared deterministic Hessian provider with GEO/Python tensor parity")
        .reference_checksum("analytic/deterministic-fixture:meshlang-hessian-provider-v1")
        .dimension("one volume one Hessian metric six symmetric tensor components")
        .formulation(
            "MeshLang provider resolves one canonical Hessian request into shared anisotropic GEO/Python fields",
        )
        .tolerance("nine provider, canonical-IR, manifest, and backend tensor checks")
        .seed("meshlang-hessian-provider-reference:cpu:v1"))
}

fn main() {
    match run() {
        Ok(result) => {
            println!(
                "QA_RESULT: {}",
                result
                    .to_json_line()
                    .expect("MeshLang Hessian result must serialize")
            );
            println!("SUCCESS");
        }
        Err(error) => {
            println!(
                "QA_RESULT: {}",
                QaResult::failed(CASE_ID, error)
                    .requested_backend("cpu")
                    .actual_backend("cpu")
                    .to_json_line()
                    .expect("MeshLang Hessian failure must serialize")
            );
            println!("FAILED");
            std::process::exit(1);
        }
    }
}
