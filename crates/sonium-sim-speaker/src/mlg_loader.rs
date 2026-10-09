//! MeshLang parameter extraction for loudspeaker models.
//!
//! Parses a `.mlg` source file and extracts `param` and `const` declarations
//! into a `LoudspeakerModel`. Only numeric scalar values are supported.

use std::collections::HashMap;

use meshlang_ast::{BinaryOp, Declaration, Expr, Literal, SourceFile, UnaryOp};

use crate::model::{
    CabinetDimensions, ConeConfig, DriverConfig, LoudspeakerModel, SeparatorConfig,
    SimulationConfig, SpeakerMaterialAssignment,
};

/// Load a `LoudspeakerModel` from a MeshLang source string.
pub fn load_model(source: &str, filename: &str) -> Result<LoudspeakerModel, String> {
    let ast = meshlang_parser::parse(source, filename).map_err(|e| format!("parse error: {e}"))?;
    let params = extract_params(&ast)?;
    build_model(&params)
}

/// Extract all `param` and `const` declarations into a name→value map.
fn extract_params(ast: &SourceFile) -> Result<HashMap<String, f64>, String> {
    let mut values: HashMap<String, f64> = HashMap::new();

    for decl in &ast.declarations {
        match decl {
            Declaration::Param(p) => {
                let v = eval_expr(&p.value, &values)?;
                let v = apply_unit(v, &p.unit);
                values.insert(p.name.name.clone(), v);
            }
            Declaration::Const(c) => {
                let v = eval_expr(&c.value, &values)?;
                let v = apply_unit(v, &c.unit);
                values.insert(c.name.name.clone(), v);
            }
            _ => {} // skip geometry, mesh, physics, export blocks
        }
    }

    Ok(values)
}

fn apply_unit(value: f64, unit: &Option<meshlang_ast::units::Unit>) -> f64 {
    match unit {
        Some(u) => value * u.to_si_factor,
        None => value,
    }
}

fn eval_expr(expr: &Expr, env: &HashMap<String, f64>) -> Result<f64, String> {
    match expr {
        Expr::Literal(literal) => match &literal.value {
            Literal::Float(v) => Ok(*v),
            Literal::Int(v) => Ok(*v as f64),
            other => Err(format!("unsupported literal: {other:?}")),
        },
        Expr::Ident(id) => env
            .get(&id.name)
            .copied()
            .ok_or_else(|| format!("undefined identifier: {}", id.name)),
        Expr::Binary(b) => {
            let l = eval_expr(&b.left, env)?;
            let r = eval_expr(&b.right, env)?;
            match b.op {
                BinaryOp::Add => Ok(l + r),
                BinaryOp::Sub => Ok(l - r),
                BinaryOp::Mul => Ok(l * r),
                BinaryOp::Div => {
                    if r.abs() < 1e-30 {
                        return Err("division by zero".into());
                    }
                    Ok(l / r)
                }
                BinaryOp::Pow => Ok(l.powf(r)),
                op => Err(format!("unsupported binary op: {op:?}")),
            }
        }
        Expr::Unary(u) => {
            let v = eval_expr(&u.operand, env)?;
            match u.op {
                UnaryOp::Neg => Ok(-v),
                op => Err(format!("unsupported unary op: {op:?}")),
            }
        }
        other => Err(format!("unsupported expression: {other:?}")),
    }
}

fn get(params: &HashMap<String, f64>, name: &str) -> Result<f64, String> {
    params
        .get(name)
        .copied()
        .ok_or_else(|| format!("missing parameter: {name}"))
}

fn get_or(params: &HashMap<String, f64>, name: &str, default: f64) -> f64 {
    params.get(name).copied().unwrap_or(default)
}

fn get_count_or(
    params: &HashMap<String, f64>,
    name: &str,
    default: usize,
) -> Result<usize, String> {
    let Some(value) = params.get(name).copied() else {
        return Ok(default);
    };
    if !value.is_finite() || value < 1.0 || value.fract() != 0.0 || value > usize::MAX as f64 {
        return Err(format!(
            "{name} must be a positive integer representable as usize, got {value}"
        ));
    }
    Ok(value as usize)
}

fn build_model(params: &HashMap<String, f64>) -> Result<LoudspeakerModel, String> {
    let cab_h = get(params, "cab_h")?;
    let cab_w = get(params, "cab_w")?;
    let cab_d = get(params, "cab_d")?;

    let woofer_r = get(params, "woofer_r")?;
    let woofer_y = get(params, "woofer_y")?;
    let tweeter_r = get(params, "tweeter_r")?;
    let tweeter_y = get(params, "tweeter_y")?;

    let piston_v = get_or(params, "piston_v", 0.001);

    let woofer_cone = match (
        params.get("woofer_cone_depth"),
        params.get("woofer_dustcap_r"),
    ) {
        (Some(&depth), Some(&dcr)) => Some(ConeConfig {
            cone_depth: depth,
            dustcap_radius: dcr,
        }),
        _ => None,
    };

    let tweeter_cone = match (
        params.get("tweeter_cone_depth"),
        params.get("tweeter_dustcap_r"),
    ) {
        (Some(&depth), Some(&dcr)) => Some(ConeConfig {
            cone_depth: depth,
            dustcap_radius: dcr,
        }),
        _ => None,
    };

    let model = LoudspeakerModel {
        cabinet: CabinetDimensions {
            height: cab_h,
            width: cab_w,
            depth: cab_d,
        },
        woofer: DriverConfig {
            radius: woofer_r,
            x_offset: 0.0,
            y_offset: woofer_y,
            piston_velocity: piston_v,
            transfer_function: None,
            driver_model: None,
            cone: woofer_cone,
        },
        tweeter: DriverConfig {
            radius: tweeter_r,
            x_offset: 0.0,
            y_offset: tweeter_y,
            piston_velocity: piston_v,
            transfer_function: None,
            driver_model: None,
            cone: tweeter_cone,
        },
        additional_drivers: Vec::new(),
        product_design: None,
        separator: SeparatorConfig {
            y_position: get_or(params, "sep_y", 0.0),
            thickness: get_or(params, "sep_t", 0.018),
        },
        waveguide: None,
        materials: SpeakerMaterialAssignment::default(),
        simulation: SimulationConfig {
            freq_start: get_or(params, "freq_start", 200.0),
            freq_end: get_or(params, "freq_end", 5000.0),
            freq_points: get_count_or(params, "freq_points", 50)?,
            speed_of_sound: get_or(params, "c_air", 343.0),
            air_density: get_or(params, "rho_air", 1.21),
            max_mesh_frequency: params.get("max_mesh_freq").copied(),
        },
    };

    model.validate()?;
    Ok(model)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_simple_params() {
        let source = r#"
            const cab_h = 0.40
            const cab_w = 0.22
            const cab_d = 0.28
            const woofer_r = 0.065
            param woofer_y = -0.08
            const tweeter_r = 0.013
            param tweeter_y = 0.10
            const sep_y = 0.02
            const sep_t = 0.018
        "#;
        let ast = meshlang_parser::parse(source, "test.mlg").unwrap();
        let params = extract_params(&ast).unwrap();
        assert!((params["cab_h"] - 0.40).abs() < 1e-10);
        assert!((params["woofer_y"] - (-0.08)).abs() < 1e-10);
        assert!((params["tweeter_y"] - 0.10).abs() < 1e-10);
    }

    #[test]
    fn test_parse_binary_expression() {
        let source = r#"
            const c_air = 343.0
            const f_max = 5000.0
            const wavelength = c_air / f_max
        "#;
        let ast = meshlang_parser::parse(source, "test.mlg").unwrap();
        let params = extract_params(&ast).unwrap();
        assert!((params["wavelength"] - 343.0 / 5000.0).abs() < 1e-10);
    }

    #[test]
    fn test_load_model_full() {
        let source = r#"
            const c_air = 343.0
            const rho_air = 1.21
            const cab_h = 0.40
            const cab_w = 0.22
            const cab_d = 0.28
            const woofer_r = 0.065
            const woofer_cone_depth = 0.008
            const woofer_dustcap_r = 0.015
            const tweeter_r = 0.013
            const tweeter_cone_depth = 0.003
            const tweeter_dustcap_r = 0.005
            const piston_v = 0.001
            param woofer_y = -0.08
            param tweeter_y = 0.10
            const sep_y = 0.02
            const sep_t = 0.018
            const freq_start = 200.0
            const freq_end = 5000.0
            const freq_points = 50
            const max_mesh_freq = 5000.0
        "#;
        let model = load_model(source, "test.mlg").unwrap();
        assert!((model.cabinet.height - 0.40).abs() < 1e-10);
        assert!((model.woofer.y_offset - (-0.08)).abs() < 1e-10);
        assert!((model.tweeter.y_offset - 0.10).abs() < 1e-10);
        assert!(model.woofer.cone.is_some());
        assert!(model.tweeter.cone.is_some());
        let wc = model.woofer.cone.unwrap();
        assert!((wc.cone_depth - 0.008).abs() < 1e-10);
        assert!((wc.dustcap_radius - 0.015).abs() < 1e-10);
        assert!(model.waveguide.is_none());
        assert_eq!(model.simulation.freq_points, 50);
    }

    #[test]
    fn test_bundled_two_way_speaker_model_is_valid() {
        let source = include_str!("../examples/two_way_speaker.mlg");
        let model = load_model(source, "two_way_speaker.mlg").unwrap();

        assert_eq!(model.simulation.freq_end, 5000.0);
        assert_eq!(model.simulation.max_mesh_frequency, Some(5000.0));
    }

    #[test]
    fn test_load_model_missing_required() {
        let source = "const cab_h = 0.40\n";
        let err = load_model(source, "test.mlg").unwrap_err();
        assert!(err.contains("missing parameter"), "got: {err}");
    }

    fn model_source_with_freq_points(freq_points: &str) -> String {
        format!(
            r#"
                const cab_h = 0.40
                const cab_w = 0.22
                const cab_d = 0.28
                const woofer_r = 0.065
                param woofer_y = -0.08
                const tweeter_r = 0.013
                param tweeter_y = 0.10
                const sep_y = 0.02
                const sep_t = 0.018
                const freq_points = {freq_points}
            "#
        )
    }

    #[test]
    fn test_load_model_rejects_fractional_frequency_count() {
        let err = load_model(&model_source_with_freq_points("3.5"), "test.mlg").unwrap_err();
        assert!(err.contains("freq_points"), "got: {err}");
    }

    #[test]
    fn test_load_model_rejects_non_positive_frequency_count() {
        for value in ["0", "-2"] {
            let err = load_model(&model_source_with_freq_points(value), "test.mlg").unwrap_err();
            assert!(err.contains("freq_points"), "value={value}, got: {err}");
        }
    }

    #[test]
    fn test_negative_param() {
        let source = r#"
            const cab_h = 0.40
            const cab_w = 0.22
            const cab_d = 0.28
            const woofer_r = 0.065
            param woofer_y = -0.08
            const tweeter_r = 0.013
            param tweeter_y = 0.10
            const sep_y = 0.02
            const sep_t = 0.018
        "#;
        let model = load_model(source, "test.mlg").unwrap();
        assert!((model.woofer.y_offset - (-0.08)).abs() < 1e-10);
    }
}
