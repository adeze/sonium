//! Wolfram golden replay: deck-text parsing.
//!
//! Replays `fixtures/funct_deck_parse_golden.json` (derived by
//! `fixtures/funct_deck_parse.wls` at 50-digit precision) through the
//! production deck parser. Regenerate with:
//! `wolframscript -script fixtures/funct_deck_parse.wls > fixtures/funct_deck_parse_golden.json`.

// Rust guideline compliant 2026-02-21

use serde_json::Value;
use sonium_fem::physics::solid::plasticity::funct_deck::{
    parse_funct_deck, parse_table0_deck, parse_table1_deck, table0_to_table2d, table1_to_table2d,
};

const GOLDEN: &str = include_str!("../fixtures/funct_deck_parse_golden.json");

const DECK: &str = "/FUNCT/10\ncurve 1\n0.0 1.0\n1.0 2.0\n\
    /FUNCT/20\ncurve 2\n0.0 1.0\n1.0 4.0\n\
    /TABLE/1/3\nrate table\n2\n10 1.0e-4\n20 1.0\n\
    /TABLE/0/9\ndata table\n2 2 2 0 0\n0.0 1.0\n0.0 10.0\n1.0 2.0 3.0 4.0\n/END\n";

fn golden_values(doc: &Value, name: &str) -> Vec<f64> {
    doc["values"][name]["values"]
        .as_array()
        .unwrap_or_else(|| panic!("golden lacks {name}"))
        .iter()
        .map(|v| {
            v.as_str()
                .unwrap_or_else(|| panic!("golden lacks {name}"))
                .parse()
                .unwrap()
        })
        .collect()
}

fn check_all(doc: &Value, name: &str, actual: &[f64]) {
    let expected = golden_values(doc, name);
    assert_eq!(actual.len(), expected.len(), "{name}: length mismatch");
    let abs: f64 = doc["tolerances"][format!("{name}_abs")].as_f64().unwrap();
    let rel: f64 = doc["tolerances"][format!("{name}_rel")].as_f64().unwrap();
    for (i, (&got, &want)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            (got - want).abs() <= abs + rel * want.abs(),
            "{name}[{i}]: production {got:.6e} vs golden {want:.6e}"
        );
    }
}

#[test]
fn funct_deck_parse_matches_wolfram() {
    let doc: Value = serde_json::from_str(GOLDEN).unwrap();
    assert_eq!(doc["meta"]["protocol"], "solids-golden-v1");
    let functs = parse_funct_deck(DECK).unwrap();
    check_all(&doc, "funct10_mid", &[functs.get(10).unwrap().eval(0.5)]);
    check_all(&doc, "funct20_mid", &[functs.get(20).unwrap().eval(0.5)]);
    let tables1 = parse_table1_deck(DECK).unwrap();
    let table1 = table1_to_table2d(&tables1[0], &functs).unwrap();
    check_all(&doc, "table1_row0", &[table1.eval(1.0, 1.0e-4)]);
    check_all(&doc, "table1_row1", &[table1.eval(1.0, 1.0)]);
    check_all(&doc, "table1_blend", &[table1.eval(0.5, 0.5)]);
    let tables0 = parse_table0_deck(DECK).unwrap();
    let table0 = table0_to_table2d(&tables0[0]).unwrap();
    check_all(&doc, "table0_center", &[table0.eval(0.5, 5.0)]);
    check_all(&doc, "table0_corner", &[table0.eval(1.0, 10.0)]);
}
