use meshlang_ast::tooling::{
    generate_tooling_artifacts, json_schema, python_bindings, typescript_bindings,
};

fn main() {
    let format = std::env::args().nth(1).unwrap_or_else(|| "all".into());
    match format.as_str() {
        "json-schema" => println!("{}", serde_json::to_string_pretty(&json_schema()).unwrap()),
        "typescript" => print!("{}", typescript_bindings()),
        "python" => print!("{}", python_bindings()),
        "all" => {
            let artifacts = generate_tooling_artifacts();
            println!("--- meshlang-ast.schema.json\n{}", artifacts.json_schema);
            println!("--- meshlang-ast.d.ts\n{}", artifacts.typescript);
            println!("--- meshlang_ast.py\n{}", artifacts.python);
        }
        _ => {
            eprintln!("usage: generate_tooling [json-schema|typescript|python|all]");
            std::process::exit(2);
        }
    }
}
