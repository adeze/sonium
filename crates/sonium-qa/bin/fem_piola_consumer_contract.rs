//! Release-facing checked H(curl)/H(div) Piola-consumer contract.

use sonium_fem::mesh::{unit_cube_tetrahedra, unit_square_triangles};
use sonium_fem::spaces::{
    FunctionSpace, HcurlSpace, HdivSpace, PiolaError, SpaceValidationError, TabulationError,
};
use sonium_qa::QaResult;

const CASE_ID: &str = "fem.piola-consumer-contract.v1";

fn main() {
    let result = match run() {
        Ok(metrics) => QaResult::passed(CASE_ID)
            .requested_backend("cpu")
            .actual_backend("cpu")
            .metric("checked_tabulations", metrics.checked_tabulations as f64)
            .minimum("checked_tabulations", 4.0)
            .metric("orientation_checks", metrics.orientation_checks as f64)
            .minimum("orientation_checks", 2.0)
            .metric("typed_rejection_checks", metrics.typed_rejection_checks as f64)
            .minimum("typed_rejection_checks", 4.0)
            .reference(
                "production H(curl)/H(div) checked tabulation, shared topology orientation, and typed singular/non-finite Jacobian rejection",
            )
            .reference_checksum("analytic/deterministic-fixture:fem-piola-consumer-contract-v1")
            .dimension("2-D triangles and 3-D tetrahedra")
            .formulation("shared covariant/contravariant/curl/divergence Piola contract")
            .tolerance("all accepted values and derivatives finite; typed rejection variants exact")
            .input_checksum("mesh:unit-square-triangles-and-unit-cube-tetrahedra:v1"),
        Err(error) => QaResult::failed(CASE_ID, error)
            .requested_backend("cpu")
            .actual_backend("cpu"),
    };

    match result.to_json_line() {
        Ok(json) => println!("QA_RESULT: {json}"),
        Err(error) => {
            eprintln!("FEM Piola consumer QA serialization failed: {error}");
            std::process::exit(1);
        }
    }

    if matches!(result.status, sonium_qa::QaStatus::Passed) {
        println!("SUCCESS");
    } else {
        println!("FAILED");
        std::process::exit(1);
    }
}

#[derive(Debug, Clone, Copy)]
struct Metrics {
    checked_tabulations: usize,
    orientation_checks: usize,
    typed_rejection_checks: usize,
}

fn run() -> Result<Metrics, String> {
    let mut checked_tabulations = 0;
    let hcurl = HcurlSpace::new();
    let hdiv = HdivSpace::new();
    let mesh_2d = unit_square_triangles(1);
    let mesh_3d = unit_cube_tetrahedra(1);

    check_finite_tabulation(
        "H(curl) 2-D",
        hcurl.try_tabulate(&mesh_2d, 0, 1.0 / 3.0, 1.0 / 3.0, 0.0),
    )?;
    checked_tabulations += 1;
    check_finite_tabulation(
        "H(div) 2-D",
        hdiv.try_tabulate(&mesh_2d, 0, 1.0 / 3.0, 1.0 / 3.0, 0.0),
    )?;
    checked_tabulations += 1;
    check_finite_tabulation(
        "H(curl) 3-D",
        hcurl.try_tabulate(&mesh_3d, 0, 0.25, 0.25, 0.25),
    )?;
    checked_tabulations += 1;
    check_finite_tabulation(
        "H(div) 3-D",
        hdiv.try_tabulate(&mesh_3d, 0, 0.25, 0.25, 0.25),
    )?;
    checked_tabulations += 1;

    let topology = sonium_fem::mesh::topology::MeshTopology::build(&mesh_2d);
    let first_edges = topology.element_edge_indices(0);
    let second_edges = topology.element_edge_indices(1);
    let (first_local, &shared_edge) = first_edges
        .iter()
        .enumerate()
        .find(|(_, edge)| second_edges.contains(edge))
        .ok_or("unit square triangles have no shared edge")?;
    let second_local = second_edges
        .iter()
        .position(|edge| *edge == shared_edge)
        .ok_or("shared edge has no second local index")?;
    if hcurl.local_to_global(&mesh_2d, 0)[first_local]
        != hcurl.local_to_global(&mesh_2d, 1)[second_local]
        || hdiv.local_to_global(&mesh_2d, 0)[first_local]
            != hdiv.local_to_global(&mesh_2d, 1)[second_local]
    {
        return Err("shared edge did not preserve one global H(curl)/H(div) DOF".into());
    }
    if topology.element_edge_signs(0)[first_local] == topology.element_edge_signs(1)[second_local] {
        return Err("shared edge orientation signs were not opposite".into());
    }

    let mut singular = unit_square_triangles(1);
    singular.nodes[3].x = 2.0;
    singular.nodes[3].y = 0.0;
    expect_error(
        "H(curl) singular 2-D",
        hcurl.try_tabulate(&singular, 0, 1.0 / 3.0, 1.0 / 3.0, 0.0),
        TabulationError::Piola(PiolaError::SingularJacobian),
    )?;
    expect_error(
        "H(div) singular 2-D",
        hdiv.try_tabulate(&singular, 0, 1.0 / 3.0, 1.0 / 3.0, 0.0),
        TabulationError::Piola(PiolaError::SingularJacobian),
    )?;

    let mut nonfinite = unit_square_triangles(1);
    nonfinite.nodes[0].x = f64::NAN;
    expect_error(
        "H(curl) non-finite 2-D",
        hcurl.try_tabulate(&nonfinite, 0, 1.0 / 3.0, 1.0 / 3.0, 0.0),
        TabulationError::Piola(PiolaError::NonFiniteJacobian),
    )?;
    expect_error(
        "H(div) non-finite 2-D",
        hdiv.try_tabulate(&nonfinite, 0, 1.0 / 3.0, 1.0 / 3.0, 0.0),
        TabulationError::Piola(PiolaError::NonFiniteJacobian),
    )?;

    let mut malformed = unit_square_triangles(1);
    malformed.elements[0].nodes.truncate(2);
    let expected = TabulationError::Space(SpaceValidationError::InvalidMesh {
        element: 0,
        reason: "element has too few node indices",
    });
    expect_error(
        "H(curl) malformed connectivity",
        hcurl.try_tabulate(&malformed, 0, 0.2, 0.2, 0.0),
        expected.clone(),
    )?;
    expect_error(
        "H(div) malformed connectivity",
        hdiv.try_tabulate(&malformed, 0, 0.2, 0.2, 0.0),
        expected,
    )?;

    Ok(Metrics {
        checked_tabulations,
        orientation_checks: 2,
        typed_rejection_checks: 6,
    })
}

fn check_finite_tabulation(
    name: &str,
    tabulation: Result<sonium_fem::spaces::BasisTabulation, TabulationError>,
) -> Result<(), String> {
    let tabulation = tabulation.map_err(|error| format!("{name} rejected valid mesh: {error}"))?;
    if tabulation
        .values
        .iter()
        .chain(tabulation.derivatives.iter())
        .any(|value| !value.is_finite())
    {
        return Err(format!("{name} produced a non-finite tabulation"));
    }
    Ok(())
}

fn expect_error<T: std::fmt::Debug>(
    name: &str,
    result: Result<T, TabulationError>,
    expected: TabulationError,
) -> Result<(), String> {
    match result {
        Err(error) if error == expected => Ok(()),
        Err(error) => Err(format!("{name} returned {error:?}, expected {expected:?}")),
        Ok(value) => Err(format!("{name} unexpectedly accepted {value:?}")),
    }
}
