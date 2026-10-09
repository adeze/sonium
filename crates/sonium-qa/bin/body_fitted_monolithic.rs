//! Bounded W2-02 body-fitted product FEM--BEM integration evidence.

use num_complex::Complex64;
use sonium_qa::QaResult;
use sonium_sim_speaker_modeling::fem_bem_coupling::MonolithicSealedProductCoupling;
use sonium_sim_speaker_modeling::{
    BraceCrossSection, BraceDesign, CabinetDesign, CabinetMaterial, CrossoverBranch,
    CrossoverNetwork, PanelAxis, PanelId, ProductDesign, ProductDriver, ProductInteriorMeshConfig,
    ProductInteriorPatchKind, build_body_fitted_sealed_product_interior_mesh,
};
use std::time::Instant;

const CASE_ID: &str = "coupling.body-fitted-monolithic.v1";

fn fixture() -> ProductDesign {
    ProductDesign {
        cabinet: CabinetDesign {
            box_width_m: 0.40,
            box_height_m: 0.30,
            box_depth_m: 0.30,
            wall_material: CabinetMaterial::Mdf18mm,
            port: None,
            baffle_width_m: 0.44,
            baffle_height_m: 0.34,
            braces: Vec::new(),
        },
        drivers: vec![ProductDriver {
            id: "w2-driver".into(),
            radius_m: 0.08,
            position_m: [0.0, 0.0, 0.0],
            normal: [0.0, 0.0, 1.0],
            force_factor_tm: 5.0,
            coil_resistance_ohm: 6.0,
            coil_reactance_ohm: 1.0,
            mechanical_resistance_n_s_per_m: 1.0,
            mechanical_reactance_n_s_per_m: 2.0,
            crossover_branch: 0,
        }],
        crossover: CrossoverNetwork {
            branches: vec![CrossoverBranch::default()],
        },
        waveguide: None,
        structural_acoustic: None,
    }
}

fn run() -> Result<(usize, usize, f64, f64, f64, f64, f64, f64, f64, u64), String> {
    let started = Instant::now();
    let design = fixture();
    let mesh = build_body_fitted_sealed_product_interior_mesh(
        &design,
        ProductInteriorMeshConfig {
            max_element_size_m: 0.025,
            min_cells_per_axis: 2,
            max_cells_per_axis: 256,
        },
    )
    .map_err(|error| format!("body-fitted mesh failed: {error}"))?;
    let node_count = mesh.diagnostics.node_count;
    let tetrahedron_count = mesh.diagnostics.tetrahedron_count;
    let coupling = MonolithicSealedProductCoupling::new(&design, mesh, 343.0, 1.21)
        .map_err(|error| format!("monolithic body-fitted coupling failed: {error}"))?;
    let solution = coupling
        .solve_frequency(&design, 120.0, Complex64::new(2.0, 0.0))
        .map_err(|error| format!("monolithic body-fitted solve failed: {error}"))?;
    if solution
        .driver_velocity_m_per_s
        .iter()
        .all(|value| value.norm() <= 1.0e-14)
        || solution
            .driver_current_a
            .iter()
            .all(|value| value.norm() <= 1.0e-14)
    {
        return Err("monolithic product solve produced no driven state".into());
    }
    let finite = [
        solution.relative_residual,
        solution.interior_relative_residual,
        solution.exterior_relative_residual,
        solution.mechanical_relative_residual,
        solution.electrical_relative_residual,
    ];
    if finite.iter().any(|value| !value.is_finite()) {
        return Err("monolithic product solve returned non-finite residual metadata".into());
    }

    // The release case also exercises the already-supported multi-wall
    // lowering. This keeps the bounded gate honest about product breadth: a
    // single front-wall driver is not the only accepted topology.
    let mut multi_wall_design = design.clone();
    let mut side_driver = multi_wall_design.drivers[0].clone();
    side_driver.id = "w2-side-driver".into();
    side_driver.radius_m = 0.03;
    side_driver.normal = [1.0, 0.0, 0.0];
    side_driver.position_m = [
        multi_wall_design.cabinet.box_width_m * 0.5,
        0.0,
        -multi_wall_design.cabinet.box_depth_m * 0.5,
    ];
    side_driver.crossover_branch = 1;
    multi_wall_design.drivers.push(side_driver);
    multi_wall_design
        .crossover
        .branches
        .push(CrossoverBranch::default());
    let multi_wall_mesh = build_body_fitted_sealed_product_interior_mesh(
        &multi_wall_design,
        ProductInteriorMeshConfig {
            max_element_size_m: 0.035,
            min_cells_per_axis: 2,
            max_cells_per_axis: 256,
        },
    )
    .map_err(|error| format!("multi-wall body-fitted mesh failed: {error}"))?;
    let multi_wall_coupling =
        MonolithicSealedProductCoupling::new(&multi_wall_design, multi_wall_mesh, 343.0, 1.21)
            .map_err(|error| format!("multi-wall monolithic coupling failed: {error}"))?;
    let multi_wall_solution = multi_wall_coupling
        .solve_frequency(&multi_wall_design, 120.0, Complex64::new(2.0, 0.0))
        .map_err(|error| format!("multi-wall monolithic solve failed: {error}"))?;
    if multi_wall_solution.driver_current_a.len() != 2
        || multi_wall_solution.driver_velocity_m_per_s.len() != 2
        || !multi_wall_solution
            .driver_current_a
            .iter()
            .all(|value| value.norm() > 0.0)
        || !multi_wall_solution
            .driver_velocity_m_per_s
            .iter()
            .all(|value| value.norm() > 0.0)
        || !multi_wall_solution.relative_residual.is_finite()
        || multi_wall_solution.relative_residual > 1.0e-7
    {
        return Err("multi-wall monolithic product solve did not close two driver blocks".into());
    }
    let mut brace_design = fixture();
    brace_design.cabinet.braces.push(BraceDesign {
        panel: PanelId::Top,
        split_along: PanelAxis::Width,
        position_fraction: 0.5,
        cross_section: Some(BraceCrossSection {
            width_m: 0.03,
            depth_m: 0.03,
        }),
    });
    let brace_mesh = build_body_fitted_sealed_product_interior_mesh(
        &brace_design,
        ProductInteriorMeshConfig {
            max_element_size_m: 0.025,
            min_cells_per_axis: 2,
            max_cells_per_axis: 256,
        },
    )
    .map_err(|error| format!("body-fitted brace mesh failed: {error}"))?;
    let brace_patch_area = brace_mesh
        .patch(ProductInteriorPatchKind::BraceWall { brace_index: 0 })
        .map(|patch| patch.area_m2)
        .filter(|area| area.is_finite() && *area > 0.0)
        .ok_or_else(|| "body-fitted brace patch is missing or empty".to_string())?;

    Ok((
        node_count,
        tetrahedron_count,
        solution.relative_residual,
        solution.interior_relative_residual,
        solution.exterior_relative_residual,
        solution.mechanical_relative_residual,
        solution.electrical_relative_residual,
        multi_wall_solution.relative_residual,
        brace_patch_area,
        started.elapsed().as_millis() as u64,
    ))
}

fn main() {
    match run() {
        Ok((
            nodes,
            tetrahedra,
            total,
            interior,
            exterior,
            mechanical,
            electrical,
            multi_wall,
            brace_patch_area,
            elapsed_ms,
        )) => {
            let result = QaResult::passed(CASE_ID)
                .requested_backend("cpu")
                .actual_backend("cpu")
                .metric("node_count", nodes as f64)
                .metric("tetrahedron_count", tetrahedra as f64)
                .metric("relative_residual", total)
                .metric("interior_relative_residual", interior)
                .metric("exterior_relative_residual", exterior)
                .metric("mechanical_relative_residual", mechanical)
                .metric("electrical_relative_residual", electrical)
                .metric("multi_wall_relative_residual", multi_wall)
                .metric("brace_patch_area_m2", brace_patch_area)
                .metric("variant_count", 3.0)
                .minimum("node_count", 1.0)
                .minimum("tetrahedron_count", 1.0)
                .maximum("relative_residual", 1.0e-7)
                .maximum("interior_relative_residual", 1.0e-7)
                .maximum("exterior_relative_residual", 1.0e-7)
                .maximum("mechanical_relative_residual", 1.0e-7)
                .maximum("electrical_relative_residual", 1.0e-7)
                .maximum("multi_wall_relative_residual", 1.0e-7)
                .minimum("brace_patch_area_m2", 1.0e-12)
                .minimum("variant_count", 3.0)
                .input_checksum("analytic/deterministic-fixture:body-fitted-monolithic-v1")
                .reference("finite driven states, two-driver residual blocks, and boundary-spanning brace patch area")
                .reference_checksum("analytic/deterministic-fixture:body-fitted-monolithic-v1")
                .dimension("3D sealed rectangular cavity; one-driver, two-driver, and braced variants; 120 Hz")
                .formulation("body-fitted sealed product interior FEM and exterior BEM monolithic solve")
                .tolerance("all residual blocks <= 1e-7; body-fitted mesh and brace patch non-empty")
                .seed("body-fitted-monolithic-v1")
                .timing_ms("body_fitted_monolithic", elapsed_ms)
                .elapsed_ms(elapsed_ms);
            println!(
                "QA_RESULT: {}",
                result
                    .to_json_line()
                    .expect("monolithic result must serialize")
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
                    .expect("monolithic failure result must serialize")
            );
            println!("FAILED");
            std::process::exit(1);
        }
    }
}
