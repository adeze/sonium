//! Mixed displacement-pressure (u-p) studies with Taylor-Hood P2-P1.
//!
//! Two plane-geometry studies share one block-assembly path: a thick-cylinder
//! annulus with exact Lamé Dirichlet data on both circles, and a
//! manufactured quadratic solution on the unit square. Both stay accurate
//! through `ν = 0.4999` (and the annulus through the exactly incompressible
//! `ν = 1/2` with one pinned pressure DOF), where P1 displacement
//! formulations lock and their solvers give up.

use crate::{StructuralExecutionProvenance, StructuralMechanicsError, StructuralModelAssumption};
use serde::{Deserialize, Serialize};
use sonium_fem::assembly::{MixedSystem, assemble_mixed_full_checked, assemble_mixed_rhs};
use sonium_fem::basis::PolynomialDegree;
use sonium_fem::forms::{LinearForm, LocalVector};
use sonium_fem::mesh::{Mesh, annular_mesh_triangles, unit_square_triangles};
use sonium_fem::physics::fluid::stokes::forms::StokesBodyForceForm;
use sonium_fem::physics::solid::mixed_up::{
    DeviatoricElasticityForm, MixedElasticMaterial, PressureComplianceForm, UpCouplingForm,
    UpCouplingTransposeForm,
};
use sonium_fem::spaces::{BasisTabulation, FunctionSpace, H1Space, VectorH1Space};
use sonium_solvers::traits::LinearOperator;
use sonium_solvers::{
    MatrixClass, SparseDirectConfig, SparseDirectMethod, TripletMatrix, factorize_sparse,
};

// Rust guideline compliant 2026-02-21

/// Zero linear form for the pressure right-hand side.
struct ZeroLinearForm;

impl LinearForm<f64> for ZeroLinearForm {
    fn integrate_point(
        &self,
        _test: &BasisTabulation,
        _weight: f64,
        _local_vector: &mut LocalVector<f64>,
    ) {
    }
}

/// Annulus geometry for the mixed u-p Lamé study.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct MixedUpAnnulusGeometry {
    /// Inner radius in m.
    pub inner_radius_m: f64,
    /// Outer radius in m.
    pub outer_radius_m: f64,
    /// Radial cell count.
    pub n_radial: usize,
    /// Angular cell count.
    pub n_angular: usize,
}

/// Isotropic material for mixed u-p studies (`ν ∈ (0, 1/2]`).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct MixedUpMaterial {
    /// Young's modulus in Pa.
    pub youngs_modulus_pa: f64,
    /// Poisson's ratio.
    pub poisson_ratio: f64,
}

/// Thick-cylinder annulus study with exact Lamé Dirichlet data.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct MixedUpAnnulusStudy {
    /// Annulus geometry.
    pub geometry: MixedUpAnnulusGeometry,
    /// Isotropic material.
    pub material: MixedUpMaterial,
    /// Internal pressure in Pa (positive pushes outward).
    pub inner_pressure_pa: f64,
    /// External pressure in Pa.
    pub outer_pressure_pa: f64,
}

impl MixedUpAnnulusStudy {
    /// Small introductory annulus.
    pub fn introductory() -> Self {
        Self {
            geometry: MixedUpAnnulusGeometry {
                inner_radius_m: 0.05,
                outer_radius_m: 0.10,
                n_radial: 6,
                n_angular: 48,
            },
            material: MixedUpMaterial {
                youngs_modulus_pa: 2.0e6,
                poisson_ratio: 0.3,
            },
            inner_pressure_pa: 1.0e5,
            outer_pressure_pa: 0.0,
        }
    }
}

/// Unit-square manufactured-solution study (quadratic exact fields).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct MixedUpMmsStudy {
    /// Cells per side.
    pub n: usize,
    /// Isotropic material (`ν < 1/2`: the manufactured field is not solenoidal).
    pub material: MixedUpMaterial,
}

impl MixedUpMmsStudy {
    /// Small introductory MMS grid.
    pub fn introductory() -> Self {
        Self {
            n: 4,
            material: MixedUpMaterial {
                youngs_modulus_pa: 2.0e6,
                poisson_ratio: 0.3,
            },
        }
    }
}

/// Result of a mixed u-p annulus solve.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MixedUpAnnulusResult {
    /// Echoed study.
    pub study: MixedUpAnnulusStudy,
    /// Displacement DOFs.
    pub displacement_dofs: usize,
    /// Pressure DOFs.
    pub pressure_dofs: usize,
    /// Relative residual of the direct saddle solve.
    pub solver_residual: f64,
    /// Whether the solve met the residual gate.
    pub converged: bool,
    /// Max nodal displacement error relative to the Lamé field.
    pub max_displacement_relative_error: f64,
    /// Max nodal pressure error relative to the exact constant pressure.
    pub max_pressure_relative_error: f64,
    /// Model assumption tag.
    pub model_assumption: StructuralModelAssumption,
    /// Execution provenance.
    pub execution: StructuralExecutionProvenance,
}

/// Result of a mixed u-p MMS solve.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MixedUpMmsResult {
    /// Echoed study.
    pub study: MixedUpMmsStudy,
    /// Displacement DOFs.
    pub displacement_dofs: usize,
    /// Pressure DOFs.
    pub pressure_dofs: usize,
    /// Relative residual of the direct saddle solve.
    pub solver_residual: f64,
    /// Whether the solve met the residual gate.
    pub converged: bool,
    /// Max nodal displacement error relative to the manufactured field.
    pub max_displacement_relative_error: f64,
    /// Max nodal pressure error relative to the manufactured pressure.
    pub max_pressure_relative_error: f64,
    /// Model assumption tag.
    pub model_assumption: StructuralModelAssumption,
    /// Execution provenance.
    pub execution: StructuralExecutionProvenance,
}

/// Plane-strain Lamé constants for the annulus.
///
/// Returns `(c1, c2, pressure)` with `u_r = c1·r + c2/r` and constant
/// pressure, evaluated through the compliance so `ν = 1/2` stays finite.
fn lame_up(
    inner_radius: f64,
    outer_radius: f64,
    inner_pressure: f64,
    outer_pressure: f64,
    material: &MixedElasticMaterial,
) -> (f64, f64, f64) {
    let ri2 = inner_radius * inner_radius;
    let ro2 = outer_radius * outer_radius;
    let area = ro2 - ri2;
    let a = (ri2 * inner_pressure - ro2 * outer_pressure) / area;
    let b = ri2 * ro2 * (inner_pressure - outer_pressure) / area;
    // c1 = A / (2(μ+λ)) and p = A·λ/(μ+λ), written via c = 1/λ.
    let c1 = if material.compliance == 0.0 {
        0.0
    } else {
        a / (2.0 * (material.mu + material.lambda))
    };
    let c2 = b / (2.0 * material.mu);
    let pressure = a / (1.0 + material.mu * material.compliance);
    (c1, c2, pressure)
}

/// Manufactured quadratic fields: `u = (x², −2xy+y²)`, `div u = 2y`,
/// `p = λ·2y`, `f = −(2μ, 4μ+2λ)`.
fn mms_fields(x: f64, y: f64, material: &MixedElasticMaterial) -> ([f64; 2], f64, [f64; 2]) {
    let displacement = [x * x, -2.0 * x * y + y * y];
    let pressure = material.lambda * 2.0 * y;
    let force = [
        -2.0 * material.mu,
        -(4.0 * material.mu + 2.0 * material.lambda),
    ];
    (displacement, pressure, force)
}

fn check_material(
    material: &MixedUpMaterial,
) -> Result<MixedElasticMaterial, StructuralMechanicsError> {
    MixedElasticMaterial::from_young_poisson(material.youngs_modulus_pa, material.poisson_ratio)
        .map_err(|error| StructuralMechanicsError::SolverFailure {
            stage: "mixed_up.material",
            message: error.to_string(),
        })
}

/// Boundary vertices and boundary edge midpoints of a 2D mesh.
///
/// Edge midpoints are included because P2 nodes on straight-sided boundary
/// edges leave curved boundaries (chords, not arcs); matching them
/// topologically avoids any geometric tolerance tuning.
fn boundary_points(mesh: &Mesh) -> (Vec<[f64; 2]>, Vec<[f64; 2]>) {
    let mut vertices = Vec::new();
    let mut midpoints = Vec::new();
    for face in &mesh.boundaries {
        for &node in &face.nodes {
            let point = &mesh.nodes[node];
            vertices.push([point.x, point.y]);
        }
        for edge in face.nodes.windows(2) {
            let a = &mesh.nodes[edge[0]];
            let b = &mesh.nodes[edge[1]];
            midpoints.push([(a.x + b.x) / 2.0, (a.y + b.y) / 2.0]);
        }
    }
    (vertices, midpoints)
}

/// Scalar-node indices lying on the mesh boundary.
///
/// Vertices match boundary vertices and edge nodes match boundary edge
/// midpoints, both at float-noise tolerance scaled to the mesh. Interior
/// diagonal midpoints stay unmatched by construction (they sit O(h) from
/// every boundary target), so no resolution-dependent tuning is needed.
fn boundary_scalar_nodes(
    coordinates: &[[f64; 3]],
    mesh: &Mesh,
    stage: &'static str,
) -> Result<Vec<usize>, StructuralMechanicsError> {
    let (vertices, midpoints) = boundary_points(mesh);
    let scale = mesh
        .nodes
        .iter()
        .map(|point| point.x.abs().max(point.y.abs()))
        .fold(1.0e-300f64, f64::max);
    let tol = 1.0e-9 * scale;
    let near = |coordinate: &[f64; 3], targets: &[[f64; 2]]| {
        targets
            .iter()
            .any(|target| (coordinate[0] - target[0]).hypot(coordinate[1] - target[1]) <= tol)
    };
    let mut matched = Vec::new();
    for (index, coordinate) in coordinates.iter().enumerate() {
        if near(coordinate, &vertices) || near(coordinate, &midpoints) {
            matched.push(index);
        }
    }
    if matched.is_empty() {
        return Err(StructuralMechanicsError::SolverFailure {
            stage,
            message: "no boundary DOFs detected".to_string(),
        });
    }
    Ok(matched)
}

/// Apply inhomogeneous Dirichlet data to the u-block rows and lift fixed
/// columns into both right-hand sides; optionally pin pressure DOF 0.
#[allow(clippy::too_many_arguments)]
fn apply_mixed_dirichlet(
    system: &MixedSystem<f64>,
    rhs_u: &mut [f64],
    rhs_p: &mut [f64],
    fixed_u: &[(usize, f64)],
    pin_pressure: bool,
) -> (TripletMatrix<f64>, Vec<f64>) {
    let n_u = system.n_dofs_0;
    let n_p = system.n_dofs_1;
    let mut prescribed_u = vec![f64::NAN; n_u];
    for &(dof, value) in fixed_u {
        prescribed_u[dof] = value;
    }
    let pinned_p = if pin_pressure { Some(0usize) } else { None };
    let mut merged = TripletMatrix::<f64>::new(n_u + n_p, n_u + n_p);

    for k in 0..system.block_00.values.len() {
        let (row, col, value) = (
            system.block_00.rows[k],
            system.block_00.cols[k],
            system.block_00.values[k],
        );
        if prescribed_u[row].is_nan() {
            if prescribed_u[col].is_nan() {
                merged.push(row, col, value);
            } else {
                rhs_u[row] -= value * prescribed_u[col];
            }
        }
    }
    for k in 0..system.block_01.values.len() {
        let (row, col, value) = (
            system.block_01.rows[k],
            system.block_01.cols[k],
            system.block_01.values[k],
        );
        if prescribed_u[row].is_nan() {
            if pinned_p == Some(col) {
                continue;
            }
            merged.push(row, n_u + col, value);
        }
    }
    for k in 0..system.block_10.values.len() {
        let (row, col, value) = (
            system.block_10.rows[k],
            system.block_10.cols[k],
            system.block_10.values[k],
        );
        if pinned_p == Some(row) {
            continue;
        }
        if prescribed_u[col].is_nan() {
            merged.push(n_u + row, col, value);
        } else {
            rhs_p[row] -= value * prescribed_u[col];
        }
    }
    for k in 0..system.block_11.values.len() {
        let (row, col, value) = (
            system.block_11.rows[k],
            system.block_11.cols[k],
            system.block_11.values[k],
        );
        if pinned_p == Some(row) || pinned_p == Some(col) {
            continue;
        }
        merged.push(n_u + row, n_u + col, value);
    }
    for (dof, &value) in prescribed_u.iter().enumerate() {
        if !value.is_nan() {
            merged.push(dof, dof, 1.0);
            rhs_u[dof] = value;
        }
    }
    if let Some(pinned) = pinned_p {
        merged.push(n_u + pinned, n_u + pinned, 1.0);
        rhs_p[pinned] = 0.0;
    }

    let mut rhs = Vec::with_capacity(n_u + n_p);
    rhs.extend_from_slice(rhs_u);
    rhs.extend_from_slice(rhs_p);
    (merged, rhs)
}

fn solve_mixed(
    merged: &TripletMatrix<f64>,
    rhs: &[f64],
    stage: &'static str,
) -> Result<(Vec<f64>, f64), StructuralMechanicsError> {
    // TripletMatrix::to_csr sorts and sums duplicates; the merged blocks
    // are not row-ordered (rows regress between blocks), so hand-rolled
    // CSR construction would corrupt the sparsity pattern.
    let operator = merged.to_csr();
    // Saddle systems carry ~1e14 block-scale disparity (stiffness vs
    // compliance mass), which defeats unequilibrated partial pivoting;
    // MC64 equilibration is required, not optional, here.
    let config = SparseDirectConfig {
        matrix_class: MatrixClass::General,
        method: SparseDirectMethod::SupernodalLu,
        equilibrate: true,
        ..SparseDirectConfig::default()
    };
    let factor = factorize_sparse(&operator, &config, None).map_err(|error| {
        StructuralMechanicsError::SolverFailure {
            stage,
            message: error.to_string(),
        }
    })?;
    let solution = factor
        .solve(rhs)
        .map_err(|error| StructuralMechanicsError::SolverFailure {
            stage,
            message: error.to_string(),
        })?;
    let residual = operator.apply(&solution);
    let mut residual_norm = 0.0;
    let mut rhs_norm = 0.0;
    for (index, &value) in residual.iter().enumerate() {
        residual_norm += (value - rhs[index]).powi(2);
        rhs_norm += rhs[index].powi(2);
    }
    Ok((solution, (residual_norm / rhs_norm.max(1.0e-300)).sqrt()))
}

/// Run the mixed u-p annulus study against the plane-strain Lamé solution.
pub fn run_mixed_up_annulus(
    study: &MixedUpAnnulusStudy,
) -> Result<MixedUpAnnulusResult, StructuralMechanicsError> {
    const STAGE: &str = "mixed_up_annulus";
    let material = check_material(&study.material)?;
    let geometry = &study.geometry;
    // NaN radii must be rejected (they would poison the mesh generator),
    // hence the explicit finite checks: plain `<=` comparisons miss NaN.
    if !geometry.inner_radius_m.is_finite()
        || !geometry.outer_radius_m.is_finite()
        || geometry.inner_radius_m <= 0.0
        || geometry.outer_radius_m <= geometry.inner_radius_m
    {
        return Err(StructuralMechanicsError::ValidationFailure {
            check: "mixed_up_annulus.radii",
            value: geometry.outer_radius_m,
        });
    }
    if !study.inner_pressure_pa.is_finite() || !study.outer_pressure_pa.is_finite() {
        return Err(StructuralMechanicsError::ValidationFailure {
            check: "mixed_up_annulus.pressures",
            value: study.inner_pressure_pa,
        });
    }
    // Direct-LU budget caps, checked before the infallible generator so its
    // internal budget panic is unreachable for admitted studies.
    if geometry.n_radial == 0 || geometry.n_radial > 32 {
        return Err(StructuralMechanicsError::ValidationFailure {
            check: "mixed_up_annulus.n_radial",
            value: geometry.n_radial as f64,
        });
    }
    if geometry.n_angular < 8 || geometry.n_angular > 256 {
        return Err(StructuralMechanicsError::ValidationFailure {
            check: "mixed_up_annulus.n_angular",
            value: geometry.n_angular as f64,
        });
    }
    let mesh = annular_mesh_triangles(
        0.0,
        0.0,
        geometry.inner_radius_m,
        geometry.outer_radius_m,
        geometry.n_radial,
        geometry.n_angular,
    );

    let u_space = VectorH1Space::new(PolynomialDegree::P2, 2);
    let p_space = H1Space::new(PolynomialDegree::P1);
    let system = assemble_mixed_full_checked(
        &DeviatoricElasticityForm::new(material.mu),
        &UpCouplingForm,
        &UpCouplingTransposeForm,
        &PressureComplianceForm::new(material.compliance),
        &u_space,
        &p_space,
        &mesh,
        4,
    )
    .map_err(|error| StructuralMechanicsError::SolverFailure {
        stage: STAGE,
        message: error.to_string(),
    })?;
    let zero_force = StokesBodyForceForm::new(|_| vec![0.0, 0.0]);
    let (mut rhs_u, mut rhs_p) =
        assemble_mixed_rhs(&zero_force, &ZeroLinearForm, &u_space, &p_space, &mesh, 4);

    let (c1, c2, pressure_exact) = lame_up(
        geometry.inner_radius_m,
        geometry.outer_radius_m,
        study.inner_pressure_pa,
        study.outer_pressure_pa,
        &material,
    );
    let exact_displacement = |x: f64, y: f64| {
        let r = x.hypot(y).max(1.0e-300);
        let radial = c1 * r + c2 / r;
        [radial * x / r, radial * y / r]
    };
    let u_coordinates =
        u_space
            .dof_coordinates(&mesh)
            .ok_or_else(|| StructuralMechanicsError::SolverFailure {
                stage: STAGE,
                message: "P2 displacement DOF coordinates unavailable".to_string(),
            })?;
    let scalar_coordinates: Vec<[f64; 3]> = u_coordinates.iter().step_by(2).copied().collect();
    let boundary_nodes = boundary_scalar_nodes(&scalar_coordinates, &mesh, STAGE)?;
    let mut fixed_u = Vec::with_capacity(2 * boundary_nodes.len());
    for &node in &boundary_nodes {
        let point = scalar_coordinates[node];
        let exact = exact_displacement(point[0], point[1]);
        fixed_u.push((2 * node, exact[0]));
        fixed_u.push((2 * node + 1, exact[1]));
    }
    let pin_pressure = material.compliance == 0.0;
    let (merged, rhs) =
        apply_mixed_dirichlet(&system, &mut rhs_u, &mut rhs_p, &fixed_u, pin_pressure);
    let (solution, residual) = solve_mixed(&merged, &rhs, STAGE)?;

    let n_u = system.n_dofs_0;
    let displacement = &solution[..n_u];
    let pressure = &solution[n_u..];
    // Annulus pressure is spatially constant, so no per-node coordinates
    // are needed; the fetch below only guards the space wiring.
    let _ =
        p_space
            .dof_coordinates(&mesh)
            .ok_or_else(|| StructuralMechanicsError::SolverFailure {
                stage: STAGE,
                message: "P1 pressure DOF coordinates unavailable".to_string(),
            })?;
    let mut max_u_error = 0.0f64;
    let mut max_u_scale = 0.0f64;
    for (node, point) in scalar_coordinates.iter().enumerate() {
        let exact = exact_displacement(point[0], point[1]);
        for component in 0..2 {
            let error = (displacement[2 * node + component] - exact[component]).abs();
            max_u_error = max_u_error.max(error);
            max_u_scale = max_u_scale.max(exact[component].abs());
        }
    }
    let mut max_p_error = 0.0f64;
    for value in pressure {
        max_p_error = max_p_error.max((value - pressure_exact).abs());
    }
    Ok(MixedUpAnnulusResult {
        study: *study,
        displacement_dofs: n_u,
        pressure_dofs: system.n_dofs_1,
        solver_residual: residual,
        converged: residual.is_finite() && residual < 1.0e-9,
        max_displacement_relative_error: max_u_error / max_u_scale.max(1.0e-300),
        max_pressure_relative_error: max_p_error / pressure_exact.abs().max(1.0e-300),
        model_assumption: StructuralModelAssumption::MixedUpElasticity,
        execution: StructuralExecutionProvenance::cpu_mixed_up_reference(),
    })
}

/// Run the mixed u-p manufactured-solution study on the unit square.
pub fn run_mixed_up_mms(
    study: &MixedUpMmsStudy,
) -> Result<MixedUpMmsResult, StructuralMechanicsError> {
    const STAGE: &str = "mixed_up_mms";
    if study.material.poisson_ratio == 0.5 {
        return Err(StructuralMechanicsError::ValidationFailure {
            check: "mixed_up_mms.poisson_ratio",
            value: study.material.poisson_ratio,
        });
    }
    let material = check_material(&study.material)?;
    if study.n == 0 || study.n > 32 {
        return Err(StructuralMechanicsError::ValidationFailure {
            check: "mixed_up_mms.resolution",
            value: study.n as f64,
        });
    }
    let mesh = unit_square_triangles(study.n);

    let u_space = VectorH1Space::new(PolynomialDegree::P2, 2);
    let p_space = H1Space::new(PolynomialDegree::P1);
    let system = assemble_mixed_full_checked(
        &DeviatoricElasticityForm::new(material.mu),
        &UpCouplingForm,
        &UpCouplingTransposeForm,
        &PressureComplianceForm::new(material.compliance),
        &u_space,
        &p_space,
        &mesh,
        4,
    )
    .map_err(|error| StructuralMechanicsError::SolverFailure {
        stage: STAGE,
        message: error.to_string(),
    })?;
    let mu = material.mu;
    let lambda = material.lambda;
    let body_force = StokesBodyForceForm::new(move |_| vec![-2.0 * mu, -(4.0 * mu + 2.0 * lambda)]);
    let (mut rhs_u, mut rhs_p) =
        assemble_mixed_rhs(&body_force, &ZeroLinearForm, &u_space, &p_space, &mesh, 4);

    let u_coordinates =
        u_space
            .dof_coordinates(&mesh)
            .ok_or_else(|| StructuralMechanicsError::SolverFailure {
                stage: STAGE,
                message: "P2 displacement DOF coordinates unavailable".to_string(),
            })?;
    let scalar_coordinates: Vec<[f64; 3]> = u_coordinates.iter().step_by(2).copied().collect();
    let boundary_nodes = boundary_scalar_nodes(&scalar_coordinates, &mesh, STAGE)?;
    let mut fixed_u = Vec::with_capacity(2 * boundary_nodes.len());
    for &node in &boundary_nodes {
        let point = scalar_coordinates[node];
        let (exact, _, _) = mms_fields(point[0], point[1], &material);
        fixed_u.push((2 * node, exact[0]));
        fixed_u.push((2 * node + 1, exact[1]));
    }
    let (merged, rhs) = apply_mixed_dirichlet(&system, &mut rhs_u, &mut rhs_p, &fixed_u, false);
    let (solution, residual) = solve_mixed(&merged, &rhs, STAGE)?;

    let n_u = system.n_dofs_0;
    let displacement = &solution[..n_u];
    let pressure = &solution[n_u..];
    let p_coordinates =
        p_space
            .dof_coordinates(&mesh)
            .ok_or_else(|| StructuralMechanicsError::SolverFailure {
                stage: STAGE,
                message: "P1 pressure DOF coordinates unavailable".to_string(),
            })?;
    let mut max_u_error = 0.0f64;
    let mut max_u_scale = 0.0f64;
    for (node, point) in scalar_coordinates.iter().enumerate() {
        let (exact, _, _) = mms_fields(point[0], point[1], &material);
        for component in 0..2 {
            let error = (displacement[2 * node + component] - exact[component]).abs();
            max_u_error = max_u_error.max(error);
            max_u_scale = max_u_scale.max(exact[component].abs());
        }
    }
    let mut max_p_error = 0.0f64;
    let mut max_p_scale = 0.0f64;
    for (node, point) in p_coordinates.iter().enumerate() {
        let (_, exact, _) = mms_fields(point[0], point[1], &material);
        max_p_error = max_p_error.max((pressure[node] - exact).abs());
        max_p_scale = max_p_scale.max(exact.abs());
    }
    Ok(MixedUpMmsResult {
        study: *study,
        displacement_dofs: n_u,
        pressure_dofs: system.n_dofs_1,
        solver_residual: residual,
        converged: residual.is_finite() && residual < 1.0e-9,
        max_displacement_relative_error: max_u_error / max_u_scale.max(1.0e-300),
        max_pressure_relative_error: max_p_error / max_p_scale.max(1.0e-300),
        model_assumption: StructuralModelAssumption::MixedUpElasticity,
        execution: StructuralExecutionProvenance::cpu_mixed_up_reference(),
    })
}
