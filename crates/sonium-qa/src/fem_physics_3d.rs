//! Shared release-facing QA for promotion of dimension-generic FEM forms.

// Rust guideline compliant 2026-02-21

use sonium_fem::assembly::{assemble_bilinear, assemble_linear};
use sonium_fem::basis::PolynomialDegree;
use sonium_fem::forms::BilinearForm;
use sonium_fem::mesh::{Mesh, unit_cube_tetrahedra, unit_square_triangles};
use sonium_fem::spaces::{FunctionSpace, H1Space};
use sonium_solvers::CsrMatrix;

use crate::QaResult;

const PATCH_TOLERANCE: f64 = 2.0e-10;

#[derive(Debug, Clone, Copy)]
struct FormMetrics {
    max_row_sum: f64,
    cross_dimension_error: f64,
    refinement_error: f64,
    energy_3d: f64,
    nnz_3d: usize,
}

/// Check a dimension-generic bilinear form's assembly invariants.
///
/// Compares prescribed-coordinate energies and row sums on tetrahedral meshes,
/// including a 2D versus z-invariant 3D comparison. No field equation is solved;
/// these contracts do not establish the named physics' solution accuracy.
pub fn run_bilinear_case<F>(case_id: &str, form: &F, constant_is_null: bool)
where
    F: BilinearForm<f64>,
{
    match bilinear_metrics(form, constant_is_null) {
        Ok(metrics) => {
            let mut record = QaResult::passed(case_id)
                .requested_backend("cpu")
                .actual_backend("cpu")
                .metric("max_row_sum", metrics.max_row_sum)
                .metric("cross_dimension_error", metrics.cross_dimension_error)
                .metric("refinement_error", metrics.refinement_error)
                .metric("energy_3d", metrics.energy_3d)
                .metric("nnz_3d", metrics.nnz_3d as f64)
                .maximum("cross_dimension_error", PATCH_TOLERANCE)
                .maximum("refinement_error", PATCH_TOLERANCE)
                .minimum("energy_3d", 0.0)
                .minimum("nnz_3d", 1.0)
                 .metric("numerical_solve_executed", 0.0)
                .maximum("numerical_solve_executed", 0.0)
                .formulation("assembly-contract only: bilinear row sums and prescribed x+y coordinate energies; no field solve or named coupling/history accuracy claim")
                .dimension("2D triangles and 3D tetrahedra; prescribed z-invariant coordinate field")
                .reference("partition of unity and prescribed-coordinate bilinear energy identities")
                .tolerance("assembly invariant relative errors <= 2e-10")
                .seed("unit-square-n2/unit-cube-tet-n1-n2:p1:x+y");
            if constant_is_null {
                record = record.maximum("max_row_sum", PATCH_TOLERANCE);
            }
            emit(record);
        }
        Err(error) => fail(case_id, &error),
    }
}

pub fn run_biharmonic_case() {
    use sonium_fem::physics::solid::biharmonic::assemble_mixed_biharmonic_operator;
    let case_id = "fem.biharmonic-3d.v1";
    let mesh_2d = unit_square_triangles(2);
    let mesh_3d = unit_cube_tetrahedra(2);
    let two = assemble_mixed_biharmonic_operator(&mesh_2d, PolynomialDegree::P1);
    let three = assemble_mixed_biharmonic_operator(&mesh_3d, PolynomialDegree::P1);
    let lower_left_nonzeros = (0..three.matrix.nnz())
        .filter(|&entry| {
            three.matrix.rows[entry] >= three.field_dofs
                && three.matrix.cols[entry] < three.field_dofs
                && three.matrix.values[entry].abs() > 1.0e-14
        })
        .count();
    if two.matrix.nnz() == 0
        || three.matrix.nnz() == 0
        || lower_left_nonzeros != 0
        || three.total_dofs() != 2 * mesh_3d.num_nodes()
    {
        fail(case_id, "mixed P1 split block contract failed");
    }
    emit(
        QaResult::passed(case_id)
            .requested_backend("cpu")
            .actual_backend("cpu")
            .metric("tetra_dofs", three.total_dofs() as f64)
            .metric("tetra_nnz", three.matrix.nnz() as f64)
            .metric("triangle_nnz", two.matrix.nnz() as f64)
            .metric("lower_left_nonzeros", lower_left_nonzeros as f64)
            .minimum("tetra_dofs", 1.0)
            .minimum("tetra_nnz", 1.0)
            .minimum("triangle_nnz", 1.0)
            .maximum("lower_left_nonzeros", 0.0)
            .metric("numerical_solve_executed",0.0).maximum("numerical_solve_executed",0.0)
            .formulation("assembly-contract only: mixed P1 biharmonic blocks; no field solve or accuracy claim")
            .reference("mixed q=-Laplacian(u) block structure; accuracy is in qa-fem-biharmonic-mms-3d")
            .seed("unit-square/unit-cube:p1:mixed-q=-laplace-u"),
    );
}

pub fn run_fsi_case() {
    use sonium_fem::physics::coupled::fsi::{
        FsiMortarConfig, FsiMortarOperators, FsiMortarSurface,
    };
    let case_id = "fem.fsi-mortar-3d.v1";
    let coarse = FsiMortarSurface::new(
        vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        vec![[0, 1, 2], [0, 2, 3]],
        vec![0, 1, 2, 3],
    )
    .unwrap_or_else(|_| fail(case_id, "coarse surface construction failed"));
    let fine = FsiMortarSurface::new(
        vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.5, 0.5, 0.0],
        ],
        vec![[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]],
        vec![0, 1, 2, 3, 4],
    )
    .unwrap_or_else(|_| fail(case_id, "fine surface construction failed"));
    let operators = FsiMortarOperators::build(&coarse, &fine, FsiMortarConfig::default())
        .unwrap_or_else(|_| fail(case_id, "nonmatching mortar construction failed"));
    let fluid = vec![1.25; coarse.trace_dofs()];
    let solid = vec![1.25; fine.trace_dofs()];
    let patch_residual = l2_norm(&operators.constraint_residual(&fluid, &solid));
    let multipliers = vec![0.75; operators.multiplier_dofs];
    let transfer = operators.multiplier_forces(&multipliers);
    let work_error = transfer.virtual_work(&fluid, &solid).abs();
    let mut moved = coarse.coordinates.clone();
    moved[0][2] += 0.08;
    let rebuild = operators
        .needs_rebuild(&moved, &fine.coordinates)
        .unwrap_or(false);
    if patch_residual > 1.0e-11 || work_error > 1.0e-11 || !rebuild {
        fail(case_id, "mortar patch/work/rebuild invariant failed");
    }
    emit(
        QaResult::passed(case_id)
            .requested_backend("cpu")
            .actual_backend("cpu")
            .metric("patch_residual", patch_residual)
            .metric("virtual_work_error", work_error)
            .metric("rebuild_triggered", rebuild as u8 as f64)
            .maximum("patch_residual", 1.0e-11)
            .maximum("virtual_work_error", 1.0e-11)
            .minimum("rebuild_triggered", 1.0)
            .metric("numerical_solve_executed",0.0).maximum("numerical_solve_executed",0.0)
            .formulation("interface-contract only: mortar patch, virtual work and rebuild; no coupled fluid/solid solve")
            .reference("constant trace and exact transpose virtual work identity; no FSI solution accuracy claim")
            .seed("coarse-2tri/fine-4tri:dual-p1:constant-vector"),
    );
}

pub fn run_compressible_case() {
    use sonium_fem::physics::fluid::compressible_flow::{
        CompressibleBoundaryCondition, CompressibleDgConfig, CompressibleDgOperator,
        CompressibleDgSolver, CompressibleDgState, ConservativeState, IdealGas, ViscosityModel,
    };
    let case_id = "fem.compressible-navier-stokes-3d.v1";
    let gas = IdealGas::AIR;
    let reference = ConservativeState::from_primitive(1.2, [25.0, 3.0, 0.0], 101_325.0, gas);
    let evaluate = |mesh: &Mesh| {
        let mut config = CompressibleDgConfig {
            viscosity: ViscosityModel::AIR,
            ..Default::default()
        };
        for face in &mesh.boundaries {
            config.boundaries.insert(
                face.marker,
                CompressibleBoundaryCondition::FarField(reference),
            );
        }
        let state = CompressibleDgState::from_function(mesh, |_| reference)
            .map_err(|error| format!("state: {error:?}"))?;
        let operator = CompressibleDgOperator::new(mesh, config.clone())
            .map_err(|error| format!("operator: {error:?}"))?;
        let residual = operator
            .spatial_operator(&state)
            .map_err(|error| format!("residual: {error:?}"))?;
        let maximum = residual
            .iter()
            .flat_map(|value| value.as_array())
            .map(f64::abs)
            .fold(0.0, f64::max);
        Ok::<_, String>((maximum, state, config))
    };
    let (residual_2d, _, _) =
        evaluate(&unit_square_triangles(2)).unwrap_or_else(|error| fail(case_id, &error));
    let mesh_3d = unit_cube_tetrahedra(1);
    let (residual_3d, mut state_3d, config_3d) =
        evaluate(&mesh_3d).unwrap_or_else(|error| fail(case_id, &error));
    let solver = CompressibleDgSolver::new(&mesh_3d, config_3d)
        .unwrap_or_else(|error| fail(case_id, &format!("solver: {error:?}")));
    let report = solver
        .step(&mut state_3d, 1.0e-6)
        .unwrap_or_else(|error| fail(case_id, &format!("step: {error:?}")));
    let viscosity_error = (ViscosityModel::AIR.dynamic_viscosity(273.15) - 1.716e-5).abs();
    if residual_2d > 1.0e-5
        || residual_3d > 1.0e-5
        || report.minimum_density <= 0.0
        || report.minimum_pressure <= 0.0
    {
        fail(case_id, "free-stream/SSPRK3/Sutherland invariant failed");
    }
    emit(
        QaResult::passed(case_id)
            .requested_backend("cpu")
            .actual_backend("cpu")
            .metric("free_stream_residual_2d", residual_2d)
            .metric("free_stream_residual_3d", residual_3d)
            .metric("minimum_density", report.minimum_density)
            .metric("minimum_pressure", report.minimum_pressure)
            .metric("sutherland_reference_error", viscosity_error)
            .maximum("free_stream_residual_2d", 1.0e-5)
            .maximum("free_stream_residual_3d", 1.0e-5)
            .maximum("sutherland_reference_error", 1.0e-12)
            .minimum("minimum_density", 1.0e-12)
            .minimum("minimum_pressure", 1.0e-12)
            .seed("uniform-flow:p1-dg:triangle/tetra:ssprk3"),
    );
}

/// Verify mixed biharmonic accuracy and refinement against a smooth manufactured solution.
///
/// Solves `u = sin(pi x) sin(pi y) sin(pi z)` on three tetrahedral meshes.
///
/// This is deliberately separate from the fast block-contract gate above: it
/// exercises source integration, essential conditions, sparse factorization,
/// the auxiliary field `q = -Delta u`, and an observed 3-D refinement rate.
pub fn run_biharmonic_mms_case() {
    use sonium_fem::physics::solid::biharmonic::{
        BiharmonicSourceForm, assemble_mixed_biharmonic_operator,
    };
    use sonium_solvers::direct::SparseLuFactorization;
    use std::f64::consts::PI;

    let case_id = "fem.biharmonic-mms-3d.v1";
    let solve = |subdivisions: usize| -> Result<(f64, f64, f64), String> {
        let mesh = unit_cube_tetrahedra(subdivisions);
        let space = H1Space::new(PolynomialDegree::P1);
        let n = space.num_global_dofs(&mesh);
        let operator = assemble_mixed_biharmonic_operator(&mesh, PolynomialDegree::P1);
        let source = BiharmonicSourceForm::new(|x, y, z| {
            9.0 * PI.powi(4) * (PI * x).sin() * (PI * y).sin() * (PI * z).sin()
        });
        let load = assemble_linear(&source, &space, &mesh, 6);
        let mut rhs = vec![0.0; 2 * n];
        rhs[n..].copy_from_slice(&load);
        let mut matrix = operator.matrix.to_csr();
        let mut constrained = Vec::new();
        for (node, point) in mesh.nodes.iter().enumerate() {
            if is_unit_cube_boundary(point.x, point.y, point.z) {
                constrained.push(node);
                constrained.push(n + node);
            }
        }
        apply_zero_dirichlet(&mut matrix, &mut rhs, &constrained)?;
        let factor = SparseLuFactorization::factorize(&matrix)
            .map_err(|error| format!("sparse factorization: {error}"))?;
        let solution = factor
            .solve(&rhs)
            .map_err(|error| format!("sparse solve: {error}"))?;
        let mut u_error = 0.0;
        let mut u_reference = 0.0;
        let mut q_error = 0.0;
        let mut q_reference = 0.0;
        // Integrate the interpolated FEM fields, including their interpolation error.
        // Order-six quadrature resolves the smooth trigonometric reference on each cell.
        let quadrature = sonium_fem::quadrature::QuadratureRule::new(
            sonium_fem::mesh::ElementType::Tetrahedron,
            6,
        );
        for element in &mesh.elements {
            let vertices = element.vertices();
            let p = vertices.iter().map(|&i| mesh.nodes[i]).collect::<Vec<_>>();
            let a = [p[1].x - p[0].x, p[1].y - p[0].y, p[1].z - p[0].z];
            let b = [p[2].x - p[0].x, p[2].y - p[0].y, p[2].z - p[0].z];
            let c = [p[3].x - p[0].x, p[3].y - p[0].y, p[3].z - p[0].z];
            let determinant = (a[0] * (b[1] * c[2] - b[2] * c[1])
                - a[1] * (b[0] * c[2] - b[2] * c[0])
                + a[2] * (b[0] * c[1] - b[1] * c[0]))
                .abs();
            for point in quadrature.iter() {
                let [xi, eta, zeta] = point.coords;
                let basis = [1.0 - xi - eta - zeta, xi, eta, zeta];
                let mut coordinates = [0.0; 3];
                let mut computed_u = 0.0;
                let mut computed_q = 0.0;
                for i in 0..4 {
                    coordinates[0] += basis[i] * p[i].x;
                    coordinates[1] += basis[i] * p[i].y;
                    coordinates[2] += basis[i] * p[i].z;
                    computed_u += basis[i] * solution[vertices[i]];
                    computed_q += basis[i] * solution[n + vertices[i]];
                }
                let exact_u = (PI * coordinates[0]).sin()
                    * (PI * coordinates[1]).sin()
                    * (PI * coordinates[2]).sin();
                let exact_q = 3.0 * PI * PI * exact_u;
                let weight = determinant * point.weight;
                u_error += weight * (computed_u - exact_u).powi(2);
                u_reference += weight * exact_u.powi(2);
                q_error += weight * (computed_q - exact_q).powi(2);
                q_reference += weight * exact_q.powi(2);
            }
        }
        let residual = relative_residual(&matrix, &solution, &rhs);
        Ok((
            (u_error / u_reference.max(1.0e-30)).sqrt(),
            (q_error / q_reference.max(1.0e-30)).sqrt(),
            residual,
        ))
    };

    // The three subdivision solves are independent (own mesh and
    // locals); run them concurrently and rejoin in order. Failure
    // semantics match the serial schedule: the first failing
    // subdivision in schedule order reports the failure.
    use rayon::prelude::*;
    let solved: Vec<(f64, f64, f64)> = [6, 12, 24]
        .into_par_iter()
        .map(solve)
        .collect::<Result<Vec<_>, _>>()
        .unwrap_or_else(|error| fail(case_id, &error));
    let [
        (u_coarse, q_coarse, residual_coarse),
        (u_medium, q_medium, residual_medium),
        (u_fine, q_fine, residual_fine),
    ] = solved
        .try_into()
        .unwrap_or_else(|_| fail(case_id, "subdivision count mismatch"));
    // Smooth P1 fields have second-order L2 convergence. Both refinement intervals
    // must reach the asymptotic regime, and the resolved field must be accurate.
    let u_convergence_rate =
        ((u_coarse / u_medium).ln() / 2.0_f64.ln()).min((u_medium / u_fine).ln() / 2.0_f64.ln());
    let q_convergence_rate =
        ((q_coarse / q_medium).ln() / 2.0_f64.ln()).min((q_medium / q_fine).ln() / 2.0_f64.ln());
    let maximum_residual = residual_coarse.max(residual_medium).max(residual_fine);
    println!(
        "Biharmonic volume L2: n6 u={u_coarse:.3e} q={q_coarse:.3e}; n12 u={u_medium:.3e} q={q_medium:.3e}; n24 u={u_fine:.3e} q={q_fine:.3e}; minimum rates u={u_convergence_rate:.3} q={q_convergence_rate:.3}"
    );
    if [
        u_coarse,
        q_coarse,
        u_medium,
        q_medium,
        u_fine,
        q_fine,
        maximum_residual,
        u_convergence_rate,
        q_convergence_rate,
    ]
    .iter()
    .any(|v| !v.is_finite())
        || u_fine > 0.02
        || q_fine > 0.01
        || u_convergence_rate < 1.7
        || q_convergence_rate < 1.7
        || maximum_residual > 1.0e-10
    {
        fail(
            case_id,
            &format!(
                "mixed biharmonic MMS convergence gate failed: u_coarse={u_coarse:.3e}, u_fine={u_fine:.3e}, q_coarse={q_coarse:.3e}, q_fine={q_fine:.3e}, u_rate={u_convergence_rate:.3}, q_rate={q_convergence_rate:.3}, residual={maximum_residual:.3e}"
            ),
        );
    }
    emit(
        QaResult::passed(case_id)
            .requested_backend("cpu")
            .actual_backend("cpu")
            .metric("coarse_u_relative_l2", u_coarse)
            .metric("coarse_q_relative_l2", q_coarse)
            .metric("medium_u_relative_l2", u_medium)
            .metric("medium_q_relative_l2", q_medium)
            .metric("fine_u_relative_l2", u_fine)
            .metric("fine_q_relative_l2", q_fine)
            .metric("u_convergence_rate", u_convergence_rate)
            .metric("q_convergence_rate", q_convergence_rate)
            .metric("maximum_algebraic_residual", maximum_residual)
            .maximum("fine_u_relative_l2", 0.02)
            .maximum("fine_q_relative_l2", 0.01)
            .minimum("u_convergence_rate", 1.7)
            .minimum("q_convergence_rate", 1.7)
            .maximum("maximum_algebraic_residual", 1.0e-10)
            .reference("u=sin(pi*x)sin(pi*y)sin(pi*z), q=3*pi^2*u, f=9*pi^4*u")
            .reference_checksum("analytic:navier-biharmonic-sin-pi-xyz:v1")
            .formulation("native mixed P1 biharmonic solve; volume-quadrature L2 accuracy and both refinement intervals")
            .metric("numerical_solve_executed", 3.0).minimum("numerical_solve_executed", 3.0)
            .seed("unit-cube-tet-n6-n12-n24:p1:mixed:sin-pi-xyz:quadrature6"),
    );
}

/// Exercise the normal-only nonmatching mortar path with an affine normal
/// field, arbitrary tangential slip, nonuniform multiplier work, explicit
/// side selection, and both sides of the reference-map rebuild threshold.
pub fn run_fsi_mortar_modes_case() {
    use sonium_fem::physics::coupled::fsi::{
        FsiMortarConfig, FsiMortarMode, FsiMortarOperators, FsiMortarSide, FsiMortarSideSelection,
        FsiMortarSurface,
    };

    let case_id = "fem.fsi-mortar-modes-3d.v1";
    let fluid = FsiMortarSurface::new(
        vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        vec![[0, 1, 2], [0, 2, 3]],
        vec![0, 1, 2, 3],
    )
    .unwrap_or_else(|_| fail(case_id, "fluid surface construction failed"));
    let solid = FsiMortarSurface::new(
        vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.5, 0.5, 0.0],
        ],
        vec![[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]],
        vec![0, 1, 2, 3, 4],
    )
    .unwrap_or_else(|_| fail(case_id, "solid surface construction failed"));
    let config = FsiMortarConfig {
        mode: FsiMortarMode::NormalOnly,
        ..FsiMortarConfig::default()
    };
    let operators = FsiMortarOperators::build(&fluid, &solid, config)
        .unwrap_or_else(|_| fail(case_id, "normal-only mortar construction failed"));
    let displacement = |point: [f64; 3], tangential: f64| {
        [
            tangential * (1.0 + point[1]),
            -0.5 * tangential * (1.0 + point[0]),
            0.2 + 0.1 * point[0] - 0.05 * point[1],
        ]
    };
    let fluid_state = fluid
        .coordinates
        .iter()
        .flat_map(|&point| displacement(point, 0.7))
        .collect::<Vec<_>>();
    let solid_state = solid
        .coordinates
        .iter()
        .flat_map(|&point| displacement(point, -1.3))
        .collect::<Vec<_>>();
    let patch_residual = l2_norm(&operators.constraint_residual(&fluid_state, &solid_state));

    let mut shifted_solid = solid_state.clone();
    for value in shifted_solid.iter_mut().skip(2).step_by(3) {
        *value += 0.01;
    }
    let residual = operators.constraint_residual(&fluid_state, &shifted_solid);
    let multipliers = (0..operators.multiplier_dofs)
        .map(|index| 0.25 + 0.1 * index as f64)
        .collect::<Vec<_>>();
    let transfer = operators.multiplier_forces(&multipliers);
    let multiplier_work = multipliers
        .iter()
        .zip(&residual)
        .map(|(lambda, gap)| lambda * gap)
        .sum::<f64>();
    let work_error = (transfer.virtual_work(&fluid_state, &shifted_solid) - multiplier_work).abs();

    let mut below = fluid.coordinates.clone();
    below[0][2] += 0.01;
    let mut above = fluid.coordinates.clone();
    above[0][2] += 0.08;
    let below_rebuild = operators
        .needs_rebuild(&below, &solid.coordinates)
        .unwrap_or_else(|_| fail(case_id, "below-threshold rebuild query failed"));
    let above_rebuild = operators
        .needs_rebuild(&above, &solid.coordinates)
        .unwrap_or_else(|_| fail(case_id, "above-threshold rebuild query failed"));
    let explicit = FsiMortarOperators::build(
        &fluid,
        &solid,
        FsiMortarConfig {
            mode: FsiMortarMode::NormalOnly,
            slave: FsiMortarSideSelection::Explicit(FsiMortarSide::Solid),
            ..FsiMortarConfig::default()
        },
    )
    .unwrap_or_else(|_| fail(case_id, "explicit-side mortar construction failed"));
    let explicit_side_ok = explicit.slave_side == FsiMortarSide::Solid;

    if patch_residual > 1.0e-11
        || work_error > 1.0e-11
        || below_rebuild
        || !above_rebuild
        || !explicit_side_ok
    {
        fail(case_id, "normal-only mortar mode/rebuild/work gate failed");
    }
    emit(
        QaResult::passed(case_id)
            .requested_backend("cpu")
            .actual_backend("cpu")
            .metric("affine_normal_patch_residual", patch_residual)
            .metric("nonuniform_virtual_work_error", work_error)
            .metric("below_threshold_rebuild", below_rebuild as u8 as f64)
            .metric("above_threshold_rebuild", above_rebuild as u8 as f64)
            .metric("explicit_solid_slave", explicit_side_ok as u8 as f64)
            .maximum("affine_normal_patch_residual", 1.0e-11)
            .maximum("nonuniform_virtual_work_error", 1.0e-11)
            .maximum("below_threshold_rebuild", 0.0)
            .minimum("above_threshold_rebuild", 1.0)
            .minimum("explicit_solid_slave", 1.0)
            .seed("coarse-2tri/fine-4tri:normal-only:affine-gap:tangential-slip"),
    );
}

/// Advance a nonuniform smooth 3-D state in a closed slip-wall box and check
/// global conservation, positivity, and that the state genuinely evolves.
pub fn run_compressible_conservation_case() {
    use sonium_fem::physics::fluid::compressible_flow::{
        CompressibleBoundaryCondition, CompressibleDgConfig, CompressibleDgSolver,
        CompressibleDgState, ConservativeState, IdealGas, ViscosityModel,
    };
    use std::f64::consts::PI;

    let case_id = "fem.compressible-conservation-3d.v1";
    let mesh = unit_cube_tetrahedra(2);
    let gas = IdealGas::AIR;
    let mut config = CompressibleDgConfig {
        viscosity: ViscosityModel::Constant(1.0e-9),
        ..CompressibleDgConfig::default()
    };
    for face in &mesh.boundaries {
        config
            .boundaries
            .insert(face.marker, CompressibleBoundaryCondition::SlipWall);
    }
    let mut state = CompressibleDgState::from_function(&mesh, |point| {
        let [x, y, z] = point;
        let density =
            1.2 * (1.0 + 0.08 * (2.0 * PI * x).sin() * (2.0 * PI * y).sin() * (2.0 * PI * z).sin());
        let velocity = [
            4.0 * (PI * x).sin() * (2.0 * PI * y).cos(),
            -3.0 * (PI * y).sin() * (2.0 * PI * z).cos(),
            2.0 * (PI * z).sin() * (2.0 * PI * x).cos(),
        ];
        let pressure = 101_325.0
            * (1.0 + 0.04 * (2.0 * PI * x).cos() * (2.0 * PI * y).cos() * (2.0 * PI * z).cos());
        ConservativeState::from_primitive(density, velocity, pressure, gas)
    })
    .unwrap_or_else(|error| fail(case_id, &format!("initial state: {error:?}")));
    let initial = state.clone();
    let solver = CompressibleDgSolver::new(&mesh, config)
        .unwrap_or_else(|error| fail(case_id, &format!("solver: {error:?}")));
    let stable_dt = solver
        .operator
        .stable_time_step(&state)
        .unwrap_or_else(|error| fail(case_id, &format!("stable time step: {error:?}")));
    let report = solver
        .step(&mut state, 0.5 * stable_dt)
        .unwrap_or_else(|error| fail(case_id, &format!("SSPRK3 step: {error:?}")));
    let relative_drift = |equation: usize| {
        (report.conserved_after[equation] - report.conserved_before[equation]).abs()
            / report.conserved_before[equation].abs().max(1.0e-30)
    };
    let mass_drift = relative_drift(0);
    let energy_drift = relative_drift(4);
    let state_change = state
        .values
        .iter()
        .zip(&initial.values)
        .map(|(after, before)| {
            after
                .as_array()
                .into_iter()
                .zip(before.as_array())
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f64>()
        })
        .sum::<f64>()
        .sqrt();
    if mass_drift > 1.0e-11
        || energy_drift > 1.0e-11
        || report.minimum_density <= 1.0e-12
        || report.minimum_pressure <= 1.0e-12
        || state_change <= 1.0e-10
    {
        fail(
            case_id,
            "nonuniform SSPRK3 conservation/positivity gate failed",
        );
    }
    emit(
        QaResult::passed(case_id)
            .requested_backend("cpu")
            .actual_backend("cpu")
            .metric("relative_mass_drift", mass_drift)
            .metric("relative_energy_drift", energy_drift)
            .metric("minimum_density", report.minimum_density)
            .metric("minimum_pressure", report.minimum_pressure)
            .metric("state_change_l2", state_change)
            .metric("accepted_time_step", report.accepted_time_step)
            .maximum("relative_mass_drift", 1.0e-11)
            .maximum("relative_energy_drift", 1.0e-11)
            .minimum("minimum_density", 1.0e-12)
            .minimum("minimum_pressure", 1.0e-12)
            .minimum("state_change_l2", 1.0e-10)
            .seed("unit-cube-tet-n2:p1-dg:smooth-3d-pulse:slip-wall:ssprk3"),
    );
}

fn bilinear_metrics<F>(form: &F, constant_is_null: bool) -> Result<FormMetrics, String>
where
    F: BilinearForm<f64>,
{
    let mesh_2d = unit_square_triangles(2);
    let mesh_3d_coarse = unit_cube_tetrahedra(1);
    let mesh_3d_fine = unit_cube_tetrahedra(2);
    let matrix_2d = assemble(form, &mesh_2d);
    let matrix_3d_coarse = assemble(form, &mesh_3d_coarse);
    let matrix_3d_fine = assemble(form, &mesh_3d_fine);
    for matrix in [&matrix_2d, &matrix_3d_coarse, &matrix_3d_fine] {
        if matrix.values.is_empty() || matrix.values.iter().any(|value| !value.is_finite()) {
            return Err("assembled matrix is empty or non-finite".into());
        }
    }
    let energy_2d = coordinate_energy(&matrix_2d, &mesh_2d);
    let energy_3d_coarse = coordinate_energy(&matrix_3d_coarse, &mesh_3d_coarse);
    let energy_3d_fine = coordinate_energy(&matrix_3d_fine, &mesh_3d_fine);
    if !energy_2d.is_finite() || !energy_3d_coarse.is_finite() || !energy_3d_fine.is_finite() {
        return Err("manufactured patch energy is non-finite".into());
    }
    let scale = energy_2d.abs().max(energy_3d_fine.abs()).max(1.0e-30);
    let cross_dimension_error = (energy_2d - energy_3d_fine).abs() / scale;
    let refinement_error =
        (energy_3d_coarse - energy_3d_fine).abs() / energy_3d_fine.abs().max(1.0e-30);
    let max_row_sum = if constant_is_null {
        matrix_3d_fine
            .row_ptrs
            .windows(2)
            .map(|range| {
                matrix_3d_fine.values[range[0]..range[1]]
                    .iter()
                    .sum::<f64>()
                    .abs()
            })
            .fold(0.0, f64::max)
    } else {
        0.0
    };
    if cross_dimension_error > PATCH_TOLERANCE
        || refinement_error > PATCH_TOLERANCE
        || (constant_is_null && max_row_sum > PATCH_TOLERANCE)
        || energy_3d_fine <= 0.0
    {
        return Err(format!(
            "patch gate failed: row={max_row_sum:.3e}, cross={cross_dimension_error:.3e}, refinement={refinement_error:.3e}, energy={energy_3d_fine:.3e}"
        ));
    }
    Ok(FormMetrics {
        max_row_sum,
        cross_dimension_error,
        refinement_error,
        energy_3d: energy_3d_fine,
        nnz_3d: matrix_3d_fine.values.len(),
    })
}

fn assemble<F>(form: &F, mesh: &Mesh) -> CsrMatrix<f64>
where
    F: BilinearForm<f64>,
{
    let space = H1Space::new(PolynomialDegree::P1);
    assemble_bilinear(form, &space, &space, mesh, 3).to_csr()
}

fn coordinate_energy(matrix: &CsrMatrix<f64>, mesh: &Mesh) -> f64 {
    let field = mesh
        .nodes
        .iter()
        .map(|point| point.x + point.y)
        .collect::<Vec<_>>();
    (0..matrix.num_rows)
        .map(|row| {
            let action = matrix
                .row_entries(row)
                .map(|(column, value)| value * field[column])
                .sum::<f64>();
            field[row] * action
        })
        .sum()
}

fn l2_norm(values: &[f64]) -> f64 {
    values.iter().map(|value| value * value).sum::<f64>().sqrt()
}

fn is_unit_cube_boundary(x: f64, y: f64, z: f64) -> bool {
    const EPSILON: f64 = 1.0e-12;
    [x, y, z]
        .into_iter()
        .any(|coordinate| coordinate.abs() < EPSILON || (coordinate - 1.0).abs() < EPSILON)
}

fn apply_zero_dirichlet(
    matrix: &mut CsrMatrix<f64>,
    rhs: &mut [f64],
    constrained_dofs: &[usize],
) -> Result<(), String> {
    let mut constrained = vec![false; matrix.num_rows];
    for &dof in constrained_dofs {
        if dof >= matrix.num_rows {
            return Err(format!("Dirichlet DOF {dof} is outside the matrix"));
        }
        constrained[dof] = true;
    }
    for row in 0..matrix.num_rows {
        for entry in matrix.row_ptrs[row]..matrix.row_ptrs[row + 1] {
            if constrained[row] || constrained[matrix.col_indices[entry]] {
                matrix.values[entry] = 0.0;
            }
        }
    }
    for &dof in constrained_dofs {
        let diagonal = (matrix.row_ptrs[dof]..matrix.row_ptrs[dof + 1])
            .find(|&entry| matrix.col_indices[entry] == dof)
            .ok_or_else(|| format!("Dirichlet DOF {dof} has no diagonal entry"))?;
        matrix.values[diagonal] = 1.0;
        rhs[dof] = 0.0;
    }
    Ok(())
}

fn relative_residual(matrix: &CsrMatrix<f64>, solution: &[f64], rhs: &[f64]) -> f64 {
    let mut numerator = 0.0;
    let mut denominator = 0.0;
    for row in 0..matrix.num_rows {
        let action = matrix
            .row_entries(row)
            .map(|(column, value)| value * solution[column])
            .sum::<f64>();
        numerator += (action - rhs[row]).powi(2);
        denominator += rhs[row] * rhs[row];
    }
    (numerator / denominator.max(1.0e-30)).sqrt()
}

fn emit(record: QaResult) {
    match record.to_json_line() {
        Ok(json) => println!("QA_RESULT: {json}"),
        Err(error) => fail("fem.3d.internal", &error),
    }
    println!("SUCCESS");
}

fn fail(case_id: &str, error: &str) -> ! {
    if let Ok(json) = QaResult::failed(case_id, error)
        .requested_backend("cpu")
        .actual_backend("cpu")
        .to_json_line()
    {
        println!("QA_RESULT: {json}");
    }
    eprintln!("FEM 3D QA FAILED: {error}");
    println!("FAILED");
    std::process::exit(1);
}
