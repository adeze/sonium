//! Matching-geometry complex viscothermal reference for a round port.

use std::collections::BTreeMap;
use std::f64::consts::TAU;

use num_complex::Complex64;
use sonium_fem::assembly::{TripletMatrix, assemble_bilinear};
use sonium_fem::basis::PolynomialDegree;
use sonium_fem::forms::{BilinearForm, LocalMatrix};
use sonium_fem::mesh::{BoundaryFace, ElementType, Mesh};
use sonium_fem::physics::acoustics::pressure::{CrossSection, EquivalentFluid, FluidProperties};
use sonium_fem::solver::{SolverConfig, SolverType, solve_csr};
use sonium_fem::spaces::{BasisTabulation, FunctionSpace, H1Space};
use sonium_solvers::{CsrMatrix, FgmresConfig, IluPreconditioner, fgmres_preconditioned};

use super::geometry3d::boundary_outward_normal;
use super::{
    MARKER_BOX_CAP, MARKER_EXTERIOR_CAP, MARKER_WALL, PortFlowError, RoundPortGeometry,
    SplineRoundPortGeometry,
};

const REFERENCE_RESIDUAL_TOLERANCE: f64 = 1.0e-8;
const DIRECT_SOLVE_MAX_DOFS: usize = 20_000;
const DIRECT_FALLBACK_MAX_DOFS: usize = 50_000;
const AUTHORITATIVE_QUADRATURE_ORDER: usize = 5;
#[cfg(test)]
const STAGE_B_P2_QUADRATURE_ORDER: usize = 5;

/// Axial radius profile backing the viscothermal linear reference.
///
/// Generalises the reference solve beyond [`RoundPortGeometry`]'s single
/// smoothstep flare: any chamber–duct–chamber profile — notably the spline
/// [`SplineRoundPortGeometry`] — supplies the local cross-section radius the
/// LRF constitutive law and the mesh-matching audit consume. All existing
/// audits (true residual ≤ 1e-8, strong/weak flux agreement, mesh-profile
/// chord tolerance) apply unchanged to every implementor.
pub trait ReferenceRadiusProfile: Clone + Send + Sync {
    /// Short geometry name used in error messages.
    fn geometry_name(&self) -> &'static str;
    /// Duct (profile) length in m.
    fn duct_length_m(&self) -> f64;
    /// Axial length of each end chamber in m.
    fn chamber_length_m(&self) -> f64;
    /// Nominal duct/bore radius in m (impedance-floor scale).
    fn duct_radius_m(&self) -> f64;
    /// Chamber radius in m.
    fn chamber_radius_m(&self) -> f64;
    /// Target mesh size in m (mesh-matching tolerance scale).
    fn mesh_size_m(&self) -> f64;
    /// Circular cross-section radius at axial coordinate `x_m`.
    fn local_radius_m(&self, x_m: f64) -> Result<f64, PortFlowError>;
    /// Whether the duct joins the chambers with sharp planar shoulders, in
    /// which case the chamber annulus is accepted on the join planes.
    fn sharp_shoulder_joins(&self) -> bool;
    /// Modelled axial domain `[x_min, x_max]` in m.
    fn axial_extent_m(&self) -> (f64, f64) {
        (
            -self.chamber_length_m(),
            self.duct_length_m() + self.chamber_length_m(),
        )
    }
}

impl ReferenceRadiusProfile for RoundPortGeometry {
    fn geometry_name(&self) -> &'static str {
        "RoundPortGeometry"
    }

    fn duct_length_m(&self) -> f64 {
        self.duct_length_m
    }

    fn chamber_length_m(&self) -> f64 {
        self.chamber_length_m
    }

    fn duct_radius_m(&self) -> f64 {
        self.duct_radius_m
    }

    fn chamber_radius_m(&self) -> f64 {
        self.chamber_radius_m
    }

    fn mesh_size_m(&self) -> f64 {
        self.mesh_size_m
    }

    fn local_radius_m(&self, x_m: f64) -> Result<f64, PortFlowError> {
        RoundPortGeometry::local_radius_m(self, x_m)
    }

    fn sharp_shoulder_joins(&self) -> bool {
        self.flare_radius_m == 0.0
    }
}

impl ReferenceRadiusProfile for SplineRoundPortGeometry {
    fn geometry_name(&self) -> &'static str {
        "SplineRoundPortGeometry"
    }

    fn duct_length_m(&self) -> f64 {
        self.duct_length_m
    }

    fn chamber_length_m(&self) -> f64 {
        self.chamber_length_m
    }

    fn duct_radius_m(&self) -> f64 {
        self.bore_radius_m
    }

    fn chamber_radius_m(&self) -> f64 {
        self.chamber_radius_m
    }

    fn mesh_size_m(&self) -> f64 {
        self.mesh_size_m
    }

    fn local_radius_m(&self, x_m: f64) -> Result<f64, PortFlowError> {
        SplineRoundPortGeometry::local_radius_m(self, x_m)
    }

    fn sharp_shoulder_joins(&self) -> bool {
        // The spline builder rejects sharp chamber-to-duct shoulders, so no
        // join plane ever carries a chamber annulus.
        false
    }
}

fn mesh_radius_m<P: ReferenceRadiusProfile>(geometry: &P, x: f64) -> Result<f64, PortFlowError> {
    let (x_min, x_max) = geometry.axial_extent_m();
    // Higher-order tetrahedron cubatures may contain points just outside the
    // reference simplex. Extend only the end chamber plateaux for those
    // assembly points; the public geometry query itself remains strictly
    // checked and mesh validation never uses this extension.
    let canonical = x.clamp(x_min, x_max);
    geometry.local_radius_m(canonical)
}

/// Physical configuration of the exact linear reference over a radius profile.
#[derive(Debug, Clone)]
pub struct LinearReferenceConfig<P> {
    /// Exact chamber/duct geometry and mesh scale used to build `mesh`.
    pub geometry: P,
    /// Equilibrium thermophysical properties used by the LRF constitutive law.
    pub fluid: FluidProperties,
    /// Lumped sealed-box volume used for the compliance transfer, in m³.
    pub box_volume_m3: f64,
}

/// Physical configuration of the exact round-port linear reference.
pub type RoundPortLinearReferenceConfig = LinearReferenceConfig<RoundPortGeometry>;

/// Physical configuration of the exact spline-port linear reference (Tier 1 oracle).
pub type SplinePortLinearReferenceConfig = LinearReferenceConfig<SplineRoundPortGeometry>;

/// Complex port impedance and box-to-port transfer at one frequency.
///
/// Every complex field is an RMS phasor using `exp(-i omega t)`. Port volume
/// velocity is positive from the box cap toward the exterior cap.
#[derive(Debug, Clone)]
pub struct ComplexPortTransfer {
    /// Harmonic frequency in Hz.
    pub frequency_hz: f64,
    /// Scalar H1 polynomial degree used for this reference solve.
    pub polynomial_degree: usize,
    /// Tetrahedral volume-quadrature polynomial order used for assembly.
    pub quadrature_order: usize,
    /// Number of scalar pressure unknowns in the reference solve.
    pub scalar_dofs: usize,
    /// Area-averaged box-cap pressure in Pa RMS.
    pub box_pressure_rms_pa: Complex64,
    /// Area-averaged exterior-cap pressure in Pa RMS.
    pub exterior_pressure_rms_pa: Complex64,
    /// Exterior-cap volume velocity in m³/s RMS, positive box-to-exterior.
    ///
    /// This authoritative conservative value is obtained from the summed
    /// uneliminated Dirichlet reactions.
    pub port_volume_velocity_rms_m3_per_s: Complex64,
    /// Strong owner-element-gradient exterior flux diagnostic in m³/s RMS.
    pub strong_exterior_volume_velocity_rms_m3_per_s: Complex64,
    /// Weak reaction flux through the box cap, positive outward from the fluid.
    pub weak_box_outward_volume_velocity_rms_m3_per_s: Complex64,
    /// Sum of weak outward box- and exterior-cap fluxes in m³/s RMS.
    ///
    /// This is the net compressible storage flow of the finite acoustic domain,
    /// rather than an incompressible zero-continuity residual.
    pub weak_net_storage_volume_velocity_rms_m3_per_s: Complex64,
    /// Relative disagreement between strong and conservative exterior flux.
    pub strong_weak_exterior_flux_relative_difference: f64,
    /// `(P_box - P_exterior) / Q_port` in Pa·s/m³.
    pub port_impedance_pa_s_per_m3: Complex64,
    /// Compliance-derived `Q_port / Q_diaphragm` transfer ratio.
    ///
    /// This is `1 / (1 - i omega C_box Z_port)` under `exp(-i omega t)`.
    pub box_to_port_flow_ratio: Complex64,
    /// Relative conditioning multiplier from complex `Z` perturbations to `H`.
    pub box_to_port_flow_ratio_condition_number: f64,
    /// True relative residual of the eliminated complex linear system.
    pub relative_residual: f64,
}

#[derive(Debug, Clone)]
struct SpatialViscothermalForm<P> {
    geometry: P,
    fluid: FluidProperties,
    omega: f64,
}

impl<P: ReferenceRadiusProfile> BilinearForm<Complex64> for SpatialViscothermalForm<P> {
    fn integrate_point(
        &self,
        test: &BasisTabulation,
        trial: &BasisTabulation,
        weight: f64,
        local_matrix: &mut LocalMatrix<Complex64>,
    ) {
        let radius = mesh_radius_m(&self.geometry, test.physical_coords[0])
            .expect("validated quadrature coordinate lies inside round-port geometry");
        let equivalent =
            EquivalentFluid::new(self.fluid.clone(), CrossSection::Circular { radius });
        let [stiffness, mass] = equivalent.prebaked_coefficients(self.omega);
        let scale = weight * test.det_j;
        for i in 0..test.n_dofs {
            for j in 0..trial.n_dofs {
                let grad_dot = (0..test.spatial_dim)
                    .map(|component| test.grad(i, component) * trial.grad(j, component))
                    .sum::<f64>();
                local_matrix.add(
                    i,
                    j,
                    (stiffness * grad_dot + mass * test.value(i) * trial.value(j)) * scale,
                );
            }
        }
    }
}

fn reference_error(message: impl Into<String>) -> PortFlowError {
    PortFlowError::LinearReference(message.into())
}

fn complex_is_finite(value: Complex64) -> bool {
    value.re.is_finite() && value.im.is_finite()
}

fn validate_config<P: ReferenceRadiusProfile>(
    config: &LinearReferenceConfig<P>,
    frequency_hz: f64,
) -> Result<(), PortFlowError> {
    config.geometry.local_radius_m(0.0)?;
    if !(frequency_hz.is_finite() && frequency_hz > 0.0) {
        return Err(reference_error(format!(
            "frequency must be positive and finite, got {frequency_hz}"
        )));
    }
    if !(config.box_volume_m3.is_finite() && config.box_volume_m3 > 0.0) {
        return Err(reference_error("box volume must be positive and finite"));
    }
    for (name, value) in [
        ("density", config.fluid.density),
        ("speed of sound", config.fluid.speed_of_sound),
        ("dynamic viscosity", config.fluid.dynamic_viscosity),
        ("thermal conductivity", config.fluid.thermal_conductivity),
        ("specific heat", config.fluid.specific_heat_cp),
        ("specific-heat ratio", config.fluid.ratio_of_specific_heats),
        ("ambient pressure", config.fluid.ambient_pressure),
    ] {
        if !(value.is_finite() && value > 0.0) {
            return Err(reference_error(format!(
                "fluid {name} must be positive and finite"
            )));
        }
    }
    Ok(())
}

fn validate_matching_mesh<P: ReferenceRadiusProfile>(
    mesh: &Mesh,
    geometry: &P,
) -> Result<(), PortFlowError> {
    if mesh.dimension != 3 {
        return Err(reference_error(format!(
            "expected a 3D mesh, got {}D",
            mesh.dimension
        )));
    }
    if mesh.elements.is_empty()
        || mesh.elements.iter().any(|element| {
            element.element_type != ElementType::Tetrahedron || element.nodes.len() != 4
        })
    {
        return Err(reference_error(
            "reference requires non-empty first-order tetrahedra",
        ));
    }
    let (x_min, x_max) = geometry.axial_extent_m();
    let tolerance = geometry.mesh_size_m() * 1.0e-6 + 1.0e-12;
    // A conforming h-refinement bisects the existing straight-sided boundary
    // triangles; its new wall nodes lie on the polyhedral chord, not exactly
    // on the analytical smoothstep profile. Permit only the generator's
    // documented small chord/profile discrepancy, never a mesh-size-scaled
    // allowance large enough to materially change the duct area.
    let radial_tolerance = |expected_radius_m: f64| 0.02 * expected_radius_m + tolerance;
    let canonical_x = |x: f64| {
        if (x - x_min).abs() <= tolerance {
            x_min
        } else if (x - x_max).abs() <= tolerance {
            x_max
        } else {
            x
        }
    };
    let mesh_min = mesh
        .nodes
        .iter()
        .map(|point| point.x)
        .fold(f64::INFINITY, f64::min);
    let mesh_max = mesh
        .nodes
        .iter()
        .map(|point| point.x)
        .fold(f64::NEG_INFINITY, f64::max);
    if (mesh_min - x_min).abs() > tolerance || (mesh_max - x_max).abs() > tolerance {
        return Err(reference_error(format!(
            "mesh axial extent does not match {}",
            geometry.geometry_name()
        )));
    }

    let mut marker_counts = BTreeMap::from([
        (MARKER_BOX_CAP, 0usize),
        (MARKER_EXTERIOR_CAP, 0usize),
        (MARKER_WALL, 0usize),
    ]);
    for (boundary_index, boundary) in mesh.boundaries.iter().enumerate() {
        if boundary.nodes.len() != 3 || boundary.nodes.iter().any(|&node| node >= mesh.nodes.len())
        {
            return Err(reference_error(format!(
                "boundary facet {boundary_index} is not a valid triangle"
            )));
        }
        let on_box_cap = boundary
            .nodes
            .iter()
            .all(|&node| (mesh.nodes[node].x - x_min).abs() <= tolerance);
        let on_exterior_cap = boundary
            .nodes
            .iter()
            .all(|&node| (mesh.nodes[node].x - x_max).abs() <= tolerance);
        let (normal, _) = boundary_outward_normal(mesh, boundary_index)
            .map_err(|error| reference_error(format!("invalid boundary topology: {error}")))?;
        let valid = match boundary.marker {
            MARKER_BOX_CAP => on_box_cap && !on_exterior_cap && normal[0] < 0.0,
            MARKER_EXTERIOR_CAP => on_exterior_cap && !on_box_cap && normal[0] > 0.0,
            MARKER_WALL => !on_box_cap && !on_exterior_cap,
            _ => false,
        };
        if !valid {
            return Err(reference_error(format!(
                "boundary facet {boundary_index} has marker {} inconsistent with the cap planes",
                boundary.marker
            )));
        }
        *marker_counts
            .get_mut(&boundary.marker)
            .expect("validated marker is present in the count map") += 1;
    }
    for marker in [MARKER_BOX_CAP, MARKER_EXTERIOR_CAP, MARKER_WALL] {
        if marker_counts[&marker] == 0 {
            return Err(reference_error(format!(
                "mesh is missing boundary marker {marker}"
            )));
        }
    }

    let mut layer_max_radius: BTreeMap<u64, f64> = BTreeMap::new();
    for point in &mesh.nodes {
        let x = canonical_x(point.x);
        // A sharp shoulder is a discontinuous cross-section. The canonical
        // profile deliberately assigns the join coordinate to the duct, but
        // the exact mesh also contains the chamber annulus on that zero-measure
        // plane. Accept the chamber radius only at the two shoulder layers.
        let at_sharp_shoulder = geometry.sharp_shoulder_joins()
            && (x.abs() <= tolerance || (x - geometry.duct_length_m()).abs() <= tolerance);
        let expected = if at_sharp_shoulder {
            geometry.chamber_radius_m()
        } else {
            geometry.local_radius_m(x)?
        };
        let radius = point.y.hypot(point.z);
        let allowed = radial_tolerance(expected);
        if !radius.is_finite() || radius > expected + allowed {
            return Err(reference_error(format!(
                "mesh radius exceeds {} profile: x={x:e}, radius={radius:e}, expected={expected:e}, excess={:e}",
                geometry.geometry_name(),
                radius - expected
            )));
        }
        layer_max_radius
            .entry(x.to_bits())
            .and_modify(|maximum| *maximum = maximum.max(radius))
            .or_insert(radius);
    }
    for (x_bits, actual) in layer_max_radius {
        let x = f64::from_bits(x_bits);
        let at_sharp_shoulder = geometry.sharp_shoulder_joins()
            && (x.abs() <= tolerance || (x - geometry.duct_length_m()).abs() <= tolerance);
        let expected = if at_sharp_shoulder {
            geometry.chamber_radius_m()
        } else {
            geometry.local_radius_m(x)?
        };
        if (actual - expected).abs() > radial_tolerance(expected) {
            return Err(reference_error(format!(
                "mesh radial extent does not match {}",
                geometry.geometry_name()
            )));
        }
    }
    Ok(())
}

fn checked_port_impedance<P: ReferenceRadiusProfile>(
    pressure_difference: Complex64,
    flow: Complex64,
    geometry: &P,
    fluid: &FluidProperties,
) -> Result<Complex64, PortFlowError> {
    if !complex_is_finite(pressure_difference) || !complex_is_finite(flow) {
        return Err(reference_error(
            "pressure difference or port flow is non-finite",
        ));
    }
    let flow_scale = pressure_difference.norm() * geometry.duct_radius_m().powi(2)
        / (fluid.density * fluid.speed_of_sound);
    if flow.norm() <= f64::EPSILON * flow_scale.max(1.0) {
        return Err(reference_error(
            "port flow is zero or too small for impedance division",
        ));
    }
    let impedance = pressure_difference / flow;
    if !complex_is_finite(impedance) {
        return Err(reference_error("port impedance is non-finite"));
    }
    Ok(impedance)
}

fn apply_complex_dirichlet(
    matrix: &TripletMatrix<Complex64>,
    rhs: &mut [Complex64],
    prescribed: &BTreeMap<usize, Complex64>,
) -> TripletMatrix<Complex64> {
    let mut eliminated = TripletMatrix::with_capacity(
        matrix.num_rows,
        matrix.num_cols,
        matrix.nnz() + prescribed.len(),
    );
    for ((&row, &col), &value) in matrix
        .rows
        .iter()
        .zip(matrix.cols.iter())
        .zip(matrix.values.iter())
    {
        if prescribed.contains_key(&row) {
            continue;
        }
        if let Some(&boundary_value) = prescribed.get(&col) {
            rhs[row] -= value * boundary_value;
        } else {
            eliminated.push(row, col, value);
        }
    }
    for (&dof, &value) in prescribed {
        eliminated.push(dof, dof, Complex64::new(1.0, 0.0));
        rhs[dof] = value;
    }
    eliminated
}

fn exact_relative_residual(
    matrix: &CsrMatrix<Complex64>,
    solution: &[Complex64],
    rhs: &[Complex64],
) -> f64 {
    let mut residual_squared = 0.0;
    let mut rhs_squared = 0.0;
    for (row, &right_hand_side) in rhs.iter().enumerate().take(matrix.num_rows) {
        let mut applied = Complex64::new(0.0, 0.0);
        for index in matrix.row_ptrs[row]..matrix.row_ptrs[row + 1] {
            applied += matrix.values[index] * solution[matrix.col_indices[index]];
        }
        residual_squared += (applied - right_hand_side).norm_sqr();
        rhs_squared += right_hand_side.norm_sqr();
    }
    residual_squared.sqrt() / rhs_squared.sqrt().max(f64::MIN_POSITIVE)
}

fn weak_reaction_volume_velocity(
    matrix: &TripletMatrix<Complex64>,
    rhs: &[Complex64],
    solution: &[Complex64],
    cap_dofs: &[usize],
    omega: f64,
) -> Result<Complex64, PortFlowError> {
    let mut reaction = vec![Complex64::new(0.0, 0.0); matrix.num_rows];
    for ((&row, &column), &value) in matrix.rows.iter().zip(&matrix.cols).zip(&matrix.values) {
        reaction[row] += value * solution[column];
    }
    for (value, forcing) in reaction.iter_mut().zip(rhs) {
        *value -= *forcing;
    }
    let summed_reaction = cap_dofs.iter().map(|&dof| reaction[dof]).sum::<Complex64>();
    let flow = Complex64::new(0.0, -1.0) * summed_reaction / omega;
    if !complex_is_finite(flow) {
        return Err(reference_error("weak cap reaction flux is non-finite"));
    }
    Ok(flow)
}

fn solve_pressure(
    matrix: &CsrMatrix<Complex64>,
    rhs: &[Complex64],
) -> Result<(Vec<Complex64>, f64), PortFlowError> {
    let solution = if matrix.num_rows <= DIRECT_SOLVE_MAX_DOFS {
        solve_csr(
            matrix,
            rhs,
            &SolverConfig {
                solver_type: SolverType::Direct,
                ..SolverConfig::default()
            },
        )
        .map_err(|error| reference_error(format!("sparse direct solve: {error}")))?
        .values
    } else {
        let preconditioner = IluPreconditioner::try_from_csr(matrix)
            .map_err(|error| reference_error(format!("ILU setup failed: {error}")))?;
        // FGMRES(50) + ILU with a sparse-direct fallback below: the small
        // restart is safe because ILU holds iteration counts low and any
        // stall falls through to the direct solve, not to a silent wrong
        // answer (SIM-PERF-20261013.md section 2, gap 4).
        let iterative = fgmres_preconditioned(
            matrix,
            &preconditioner,
            rhs,
            None,
            &FgmresConfig {
                max_iterations: 500,
                restart: 50,
                tolerance: 1.0e-10,
                print_interval: 0,
                convergence_check_interval: 1,
                exact_residual_check_interval: 1,
            },
        );
        let residual = exact_relative_residual(matrix, &iterative.x, rhs);
        if iterative.converged && residual <= REFERENCE_RESIDUAL_TOLERANCE {
            iterative.x
        } else if matrix.num_rows <= DIRECT_FALLBACK_MAX_DOFS {
            solve_csr(
                matrix,
                rhs,
                &SolverConfig {
                    solver_type: SolverType::Direct,
                    ..SolverConfig::default()
                },
            )
            .map_err(|error| {
                reference_error(format!("iterative and fallback solve failed: {error}"))
            })?
            .values
        } else {
            return Err(reference_error(format!(
                "FGMRES+ILU did not converge: residual={residual:e}"
            )));
        }
    };
    if solution.iter().any(|&value| !complex_is_finite(value)) {
        return Err(reference_error("solver returned a non-finite pressure"));
    }
    let residual = exact_relative_residual(matrix, &solution, rhs);
    if !(residual.is_finite() && residual <= REFERENCE_RESIDUAL_TOLERANCE) {
        return Err(reference_error(format!(
            "true relative residual {residual:e} exceeds 1e-8"
        )));
    }
    Ok((solution, residual))
}

fn face_quadrature_reference_points(
    mesh: &Mesh,
    boundary: &BoundaryFace,
) -> Result<[[f64; 3]; 3], PortFlowError> {
    let owner = mesh.elements.get(boundary.element_idx).ok_or_else(|| {
        reference_error(format!(
            "boundary references missing owner {}",
            boundary.element_idx
        ))
    })?;
    if owner.element_type != ElementType::Tetrahedron
        || owner.nodes.len() < 4
        || boundary.nodes.len() != 3
    {
        return Err(reference_error(
            "cap quadrature requires a triangular tetrahedron face",
        ));
    }
    let reference_vertices = [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ];
    let mut face_reference_vertices = [[0.0; 3]; 3];
    for (face_local, &node) in boundary.nodes.iter().enumerate() {
        let owner_local = owner.nodes[..4]
            .iter()
            .position(|&candidate| candidate == node)
            .ok_or_else(|| reference_error("cap node is not owned by its tetrahedron"))?;
        face_reference_vertices[face_local] = reference_vertices[owner_local];
    }

    // Symmetric degree-two triangle rule. Each point has physical weight A/3.
    let barycentric = [
        [2.0 / 3.0, 1.0 / 6.0, 1.0 / 6.0],
        [1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0],
        [1.0 / 6.0, 1.0 / 6.0, 2.0 / 3.0],
    ];
    Ok(barycentric.map(|weights| {
        let mut point = [0.0; 3];
        for face_local in 0..3 {
            for component in 0..3 {
                point[component] +=
                    weights[face_local] * face_reference_vertices[face_local][component];
            }
        }
        point
    }))
}

fn cap_average(
    space: &H1Space,
    mesh: &Mesh,
    pressure: &[Complex64],
    marker: i32,
) -> Result<Complex64, PortFlowError> {
    let mut weighted = Complex64::new(0.0, 0.0);
    let mut area_sum = 0.0;
    for (index, boundary) in mesh
        .boundaries
        .iter()
        .enumerate()
        .filter(|(_, b)| b.marker == marker)
    {
        let (_, area) = boundary_outward_normal(mesh, index)?;
        let dofs = space.local_to_global(mesh, boundary.element_idx);
        let mut face_average = Complex64::new(0.0, 0.0);
        for point in face_quadrature_reference_points(mesh, boundary)? {
            let tabulation =
                space.tabulate(mesh, boundary.element_idx, point[0], point[1], point[2]);
            face_average += dofs
                .iter()
                .enumerate()
                .map(|(local, &dof)| pressure[dof] * tabulation.value(local))
                .sum::<Complex64>()
                / 3.0;
        }
        weighted += face_average * area;
        area_sum += area;
    }
    if !(area_sum.is_finite() && area_sum > 0.0) {
        return Err(reference_error(format!(
            "marker {marker} has zero cap area"
        )));
    }
    Ok(weighted / area_sum)
}

fn exterior_volume_velocity<P: ReferenceRadiusProfile>(
    space: &H1Space,
    mesh: &Mesh,
    pressure: &[Complex64],
    geometry: &P,
    fluid: &FluidProperties,
    omega: f64,
) -> Result<Complex64, PortFlowError> {
    let mut flow = Complex64::new(0.0, 0.0);
    for (boundary_index, boundary) in mesh
        .boundaries
        .iter()
        .enumerate()
        .filter(|(_, boundary)| boundary.marker == MARKER_EXTERIOR_CAP)
    {
        let (normal, area) = boundary_outward_normal(mesh, boundary_index)?;
        let dofs = space.local_to_global(mesh, boundary.element_idx);
        let mut face_flow = Complex64::new(0.0, 0.0);
        for point in face_quadrature_reference_points(mesh, boundary)? {
            let tabulation =
                space.tabulate(mesh, boundary.element_idx, point[0], point[1], point[2]);
            let mut pressure_gradient = [Complex64::new(0.0, 0.0); 3];
            for (local, &dof) in dofs.iter().enumerate() {
                for (component, slot) in pressure_gradient.iter_mut().enumerate() {
                    *slot += pressure[dof] * tabulation.grad(local, component);
                }
            }
            let radius = mesh_radius_m(geometry, tabulation.physical_coords[0])?;
            let rho_eff = EquivalentFluid::new(fluid.clone(), CrossSection::Circular { radius })
                .effective_density(omega);
            let normal_gradient = (0..3)
                .map(|component| pressure_gradient[component] * normal[component])
                .sum::<Complex64>();
            face_flow += normal_gradient / rho_eff / 3.0;
        }
        flow += Complex64::new(0.0, -1.0) * face_flow * area / omega;
    }
    if !complex_is_finite(flow) {
        return Err(reference_error("exterior cap flux is non-finite"));
    }
    Ok(flow)
}

fn solve_with_degree_and_quadrature_order<P: ReferenceRadiusProfile>(
    mesh: &Mesh,
    config: &LinearReferenceConfig<P>,
    frequency_hz: f64,
    degree: PolynomialDegree,
    quadrature_order: usize,
) -> Result<ComplexPortTransfer, PortFlowError> {
    validate_config(config, frequency_hz)?;
    validate_matching_mesh(mesh, &config.geometry)?;
    let omega = TAU * frequency_hz;
    let space = H1Space::new(degree);
    let form = SpatialViscothermalForm {
        geometry: config.geometry.clone(),
        fluid: config.fluid.clone(),
        omega,
    };
    let original_matrix = assemble_bilinear(&form, &space, &space, mesh, quadrature_order);
    let original_rhs = vec![Complex64::new(0.0, 0.0); space.num_global_dofs(mesh)];
    let box_dofs = space.boundary_dofs(mesh, MARKER_BOX_CAP);
    let exterior_dofs = space.boundary_dofs(mesh, MARKER_EXTERIOR_CAP);
    let mut prescribed = BTreeMap::new();
    for &dof in &box_dofs {
        prescribed.insert(dof, Complex64::new(1.0, 0.0));
    }
    for &dof in &exterior_dofs {
        prescribed.insert(dof, Complex64::new(0.0, 0.0));
    }
    if prescribed.is_empty() {
        return Err(reference_error("cap Dirichlet sets are empty"));
    }
    let mut rhs = vec![Complex64::new(0.0, 0.0); space.num_global_dofs(mesh)];
    let matrix = apply_complex_dirichlet(&original_matrix, &mut rhs, &prescribed).to_csr();
    let (pressure, relative_residual) = solve_pressure(&matrix, &rhs)?;
    let box_pressure = cap_average(&space, mesh, &pressure, MARKER_BOX_CAP)?;
    let exterior_pressure = cap_average(&space, mesh, &pressure, MARKER_EXTERIOR_CAP)?;
    let flow = weak_reaction_volume_velocity(
        &original_matrix,
        &original_rhs,
        &pressure,
        &exterior_dofs,
        omega,
    )?;
    let box_outward_flow = weak_reaction_volume_velocity(
        &original_matrix,
        &original_rhs,
        &pressure,
        &box_dofs,
        omega,
    )?;
    let strong_flow = exterior_volume_velocity(
        &space,
        mesh,
        &pressure,
        &config.geometry,
        &config.fluid,
        omega,
    )?;
    let strong_weak_difference = (strong_flow - flow).norm() / flow.norm().max(f64::EPSILON);
    let pressure_difference = box_pressure - exterior_pressure;
    let impedance =
        checked_port_impedance(pressure_difference, flow, &config.geometry, &config.fluid)?;
    let compliance =
        config.box_volume_m3 / (config.fluid.density * config.fluid.speed_of_sound.powi(2));
    let transfer = Complex64::new(1.0, 0.0)
        / (Complex64::new(1.0, 0.0) - Complex64::new(0.0, omega * compliance) * impedance);
    let transfer_condition =
        (Complex64::new(0.0, omega * compliance) * impedance * transfer).norm();
    if [
        box_pressure,
        exterior_pressure,
        flow,
        strong_flow,
        impedance,
        transfer,
    ]
    .into_iter()
    .any(|value| !complex_is_finite(value))
    {
        return Err(reference_error(
            "derived complex reference quantity is non-finite",
        ));
    }
    Ok(ComplexPortTransfer {
        frequency_hz,
        polynomial_degree: space.polynomial_degree(),
        quadrature_order,
        scalar_dofs: space.num_global_dofs(mesh),
        box_pressure_rms_pa: box_pressure,
        exterior_pressure_rms_pa: exterior_pressure,
        port_volume_velocity_rms_m3_per_s: flow,
        strong_exterior_volume_velocity_rms_m3_per_s: strong_flow,
        weak_box_outward_volume_velocity_rms_m3_per_s: box_outward_flow,
        weak_net_storage_volume_velocity_rms_m3_per_s: box_outward_flow + flow,
        strong_weak_exterior_flux_relative_difference: strong_weak_difference,
        port_impedance_pa_s_per_m3: impedance,
        box_to_port_flow_ratio: transfer,
        box_to_port_flow_ratio_condition_number: transfer_condition,
        relative_residual,
    })
}

#[cfg(test)]
fn solve_p1_with_quadrature_order<P: ReferenceRadiusProfile>(
    mesh: &Mesh,
    config: &LinearReferenceConfig<P>,
    frequency_hz: f64,
    quadrature_order: usize,
) -> Result<ComplexPortTransfer, PortFlowError> {
    solve_with_degree_and_quadrature_order(
        mesh,
        config,
        frequency_hz,
        PolynomialDegree::P1,
        quadrature_order,
    )
}

/// Solve the complex LRF pressure field on the supplied exact round-port mesh.
///
/// The returned RMS phasors use `exp(-i omega t)`, with port flow positive
/// from the box cap toward the exterior cap. The supplied mesh must be the
/// complete, marker-preserving 3D tetrahedral mesh for `config.geometry`.
/// The authoritative discretization is scalar quadratic H1 (tetrahedral P2)
/// with the interior degree-5 tetrahedral cubature. The returned provenance
/// records this degree, quadrature order, and scalar pressure-DOF count.
///
/// # Errors
///
/// Returns [`PortFlowError::LinearReference`] for invalid physical inputs,
/// mismatched geometry or boundary topology, assembly/solver failure, a true
/// relative residual above `1e-8`, non-finite fields, or flow too small to
/// form the port impedance.
pub fn solve_round_port_linear_reference(
    mesh: &Mesh,
    config: &RoundPortLinearReferenceConfig,
    frequency_hz: f64,
) -> Result<ComplexPortTransfer, PortFlowError> {
    solve_with_degree_and_quadrature_order(
        mesh,
        config,
        frequency_hz,
        PolynomialDegree::P2,
        AUTHORITATIVE_QUADRATURE_ORDER,
    )
}

/// Solve the complex LRF pressure field on the supplied spline-port mesh.
///
/// Same Tier 1 oracle as [`solve_round_port_linear_reference`] with the duct
/// wall taken from the [`SplinePortProfile`](super::profile::SplinePortProfile)
/// carried by `config.geometry` instead of the single smoothstep flare. The
/// supplied mesh must be the complete, marker-preserving 3D tetrahedral mesh
/// for `config.geometry` (see
/// [`build_spline_round_port_mesh`](super::geometry3d::build_spline_round_port_mesh));
/// the mesh-profile audit, the true-residual ≤ 1e-8 gate, and the strong/weak
/// flux diagnostics apply unchanged. Returned RMS phasors use `exp(-i omega t)`
/// with port flow positive from the box cap toward the exterior cap.
///
/// # Errors
///
/// Returns [`PortFlowError::LinearReference`] for invalid physical inputs,
/// mismatched geometry or boundary topology, assembly/solver failure, a true
/// relative residual above `1e-8`, non-finite fields, or flow too small to
/// form the port impedance.
pub fn solve_spline_port_linear_reference(
    mesh: &Mesh,
    config: &SplinePortLinearReferenceConfig,
    frequency_hz: f64,
) -> Result<ComplexPortTransfer, PortFlowError> {
    solve_with_degree_and_quadrature_order(
        mesh,
        config,
        frequency_hz,
        PolynomialDegree::P2,
        AUTHORITATIVE_QUADRATURE_ORDER,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use sonium_fem::mesh::{BoundaryFace, BoundaryType, Element, Point};

    fn assert_complex_close(actual: Complex64, expected: Complex64, tolerance: f64) {
        assert!(
            (actual - expected).norm() < tolerance,
            "{actual:?} != {expected:?}"
        );
    }

    #[test]
    fn complex_dirichlet_elimination_corrects_columns_and_preserves_symmetry() {
        let mut matrix = TripletMatrix::new(3, 3);
        let values = [
            [
                Complex64::new(4.0, 0.0),
                Complex64::new(1.0, 1.0),
                Complex64::new(0.0, 0.0),
            ],
            [
                Complex64::new(1.0, -1.0),
                Complex64::new(5.0, 0.0),
                Complex64::new(2.0, 0.0),
            ],
            [
                Complex64::new(0.0, 0.0),
                Complex64::new(2.0, 0.0),
                Complex64::new(6.0, 0.0),
            ],
        ];
        for (row, entries) in values.iter().enumerate() {
            for (col, &value) in entries.iter().enumerate() {
                matrix.push(row, col, value);
            }
        }
        let boundary_value = Complex64::new(2.0, -0.5);
        let prescribed = BTreeMap::from([(0, boundary_value)]);
        let mut rhs = vec![Complex64::new(1.0, 0.0); 3];
        let eliminated = apply_complex_dirichlet(&matrix, &mut rhs, &prescribed).to_csr();
        assert_complex_close(
            rhs[1],
            Complex64::new(1.0, 0.0) - values[1][0] * boundary_value,
            1e-14,
        );
        assert_eq!(rhs[0], boundary_value);
        assert_eq!(
            eliminated.row_entries(0).collect::<Vec<_>>(),
            vec![(0, Complex64::new(1.0, 0.0))]
        );
        assert!(eliminated.row_entries(1).all(|(column, _)| column != 0));
    }

    #[test]
    fn iterative_reference_reports_checked_ilu_setup_failure() {
        let n = DIRECT_SOLVE_MAX_DOFS + 1;
        let matrix = CsrMatrix::from_raw_parts(n, n, vec![0; n + 1], Vec::new(), Vec::new());
        let rhs = vec![Complex64::new(1.0, 0.0); n];
        let error = solve_pressure(&matrix, &rhs).expect_err("missing ILU pivots must fail setup");
        assert!(matches!(
            error,
            PortFlowError::LinearReference(message) if message.contains("ILU setup failed")
        ));
    }

    #[test]
    fn weak_reaction_flux_has_exp_minus_iwt_sign_and_conserves_cap_sum() {
        let mut matrix = TripletMatrix::new(3, 3);
        for (row, column, value) in [
            (0, 0, Complex64::new(2.0, 0.0)),
            (0, 1, Complex64::new(-1.0, 0.0)),
            (1, 0, Complex64::new(-1.0, 0.0)),
            (1, 1, Complex64::new(2.0, 0.0)),
            (1, 2, Complex64::new(-1.0, 0.0)),
            (2, 1, Complex64::new(-1.0, 0.0)),
            (2, 2, Complex64::new(2.0, 0.0)),
        ] {
            matrix.push(row, column, value);
        }
        let pressure = [
            Complex64::new(1.0, 0.0),
            Complex64::new(0.5, 0.0),
            Complex64::new(0.0, 0.0),
        ];
        let rhs = [Complex64::new(0.0, 0.0); 3];
        let omega = 2.0;
        let box_flow =
            weak_reaction_volume_velocity(&matrix, &rhs, &pressure, &[0], omega).unwrap();
        let exterior_flow =
            weak_reaction_volume_velocity(&matrix, &rhs, &pressure, &[2], omega).unwrap();
        assert_complex_close(box_flow, Complex64::new(0.0, -0.75), 1e-14);
        assert_complex_close(exterior_flow, Complex64::new(0.0, 0.25), 1e-14);
        let both = weak_reaction_volume_velocity(&matrix, &rhs, &pressure, &[0, 2], omega).unwrap();
        assert_complex_close(both, box_flow + exterior_flow, 1e-14);
    }

    fn single_tetra_mesh() -> Mesh {
        let mut mesh = Mesh::new(3);
        for point in [
            Point::new_3d(0.0, 0.0, 0.0),
            Point::new_3d(1.0, 0.0, 0.0),
            Point::new_3d(0.0, 1.0, 0.0),
            Point::new_3d(0.0, 0.0, 1.0),
        ] {
            mesh.add_node(point);
        }
        mesh.elements
            .push(Element::new(ElementType::Tetrahedron, vec![0, 1, 2, 3], 0));
        mesh.boundaries = vec![
            BoundaryFace {
                nodes: vec![0, 2, 3],
                boundary_type: BoundaryType::Dirichlet,
                marker: MARKER_BOX_CAP,
                element_idx: 0,
                local_idx: 2,
            },
            BoundaryFace {
                nodes: vec![1, 2, 3],
                boundary_type: BoundaryType::Dirichlet,
                marker: MARKER_EXTERIOR_CAP,
                element_idx: 0,
                local_idx: 3,
            },
            BoundaryFace {
                nodes: vec![0, 1, 3],
                boundary_type: BoundaryType::Neumann,
                marker: MARKER_WALL,
                element_idx: 0,
                local_idx: 1,
            },
            BoundaryFace {
                nodes: vec![0, 1, 2],
                boundary_type: BoundaryType::Neumann,
                marker: MARKER_WALL,
                element_idx: 0,
                local_idx: 0,
            },
        ];
        mesh
    }

    #[test]
    fn cap_average_and_manufactured_linear_flux_are_exact() {
        let mesh = single_tetra_mesh();
        let space = H1Space::new(PolynomialDegree::P1);
        let constant = Complex64::new(2.0, -3.0);
        assert_complex_close(
            cap_average(&space, &mesh, &[constant; 4], MARKER_BOX_CAP).unwrap(),
            constant,
            1e-14,
        );
        assert_complex_close(
            cap_average(&space, &mesh, &[constant; 4], MARKER_EXTERIOR_CAP).unwrap(),
            constant,
            1e-14,
        );

        let geometry = RoundPortGeometry {
            duct_length_m: 1.0,
            duct_radius_m: 2.0,
            chamber_length_m: 1.0,
            chamber_radius_m: 2.0,
            flare_radius_m: 0.0,
            mesh_size_m: 1.0,
        };
        let fluid = FluidProperties::air_20c();
        let omega = 100.0;
        let pressure = mesh
            .nodes
            .iter()
            .map(|point| Complex64::new(1.0, 2.0) * point.x)
            .collect::<Vec<_>>();
        let flow =
            exterior_volume_velocity(&space, &mesh, &pressure, &geometry, &fluid, omega).unwrap();
        let (_, area) = boundary_outward_normal(&mesh, 1).unwrap();
        let rho_eff = EquivalentFluid::new(fluid, CrossSection::Circular { radius: 2.0 })
            .effective_density(omega);
        let expected =
            Complex64::new(0.0, -1.0) * Complex64::new(1.0, 2.0) * (1.0 / 3.0_f64.sqrt()) * area
                / (omega * rho_eff);
        assert_complex_close(flow, expected, 1e-14);
    }

    #[test]
    fn p2_cap_average_and_quadratic_strong_flux_use_edge_dofs() {
        let mesh = single_tetra_mesh();
        let space = H1Space::new(PolynomialDegree::P2);
        let pressure = space
            .dof_coordinates(&mesh)
            .unwrap()
            .into_iter()
            .map(|point| Complex64::new(point[0].powi(2), 0.0))
            .collect::<Vec<_>>();

        // On x+y+z=1, x is a barycentric coordinate and <x^2> = 1/6.
        let average = cap_average(&space, &mesh, &pressure, MARKER_EXTERIOR_CAP).unwrap();
        assert_complex_close(average, Complex64::new(1.0 / 6.0, 0.0), 1e-14);

        let geometry = RoundPortGeometry {
            duct_length_m: 1.0,
            duct_radius_m: 2.0,
            chamber_length_m: 1.0,
            chamber_radius_m: 2.0,
            flare_radius_m: 0.0,
            mesh_size_m: 1.0,
        };
        let fluid = FluidProperties::air_20c();
        let omega = 100.0;
        let flow =
            exterior_volume_velocity(&space, &mesh, &pressure, &geometry, &fluid, omega).unwrap();
        let (_, area) = boundary_outward_normal(&mesh, 1).unwrap();
        let rho_eff = EquivalentFluid::new(fluid, CrossSection::Circular { radius: 2.0 })
            .effective_density(omega);
        // grad(x^2)·(1,1,1)/sqrt(3) averaged on the face is 2/(3 sqrt(3)).
        let expected =
            Complex64::new(0.0, -1.0) * (2.0 / (3.0 * 3.0_f64.sqrt())) * area / (omega * rho_eff);
        assert_complex_close(flow, expected, 1e-14);
    }

    #[test]
    fn p2_cap_reaction_includes_all_three_edge_midpoints() {
        let mesh = single_tetra_mesh();
        let space = H1Space::new(PolynomialDegree::P2);
        let cap_dofs = space.boundary_dofs(&mesh, MARKER_BOX_CAP);
        assert_eq!(cap_dofs.len(), 6);
        assert_eq!(
            cap_dofs
                .iter()
                .filter(|&&dof| dof < mesh.num_nodes())
                .count(),
            3
        );
        assert_eq!(
            cap_dofs
                .iter()
                .filter(|&&dof| dof >= mesh.num_nodes())
                .count(),
            3
        );

        let n = space.num_global_dofs(&mesh);
        let mut matrix = TripletMatrix::new(n, n);
        for dof in 0..n {
            matrix.push(dof, dof, Complex64::new(1.0, 0.0));
        }
        let mut pressure = vec![Complex64::new(0.0, 0.0); n];
        for &dof in &cap_dofs {
            if dof >= mesh.num_nodes() {
                pressure[dof] = Complex64::new(1.0, 0.0);
            }
        }
        let rhs = vec![Complex64::new(0.0, 0.0); n];
        let flow = weak_reaction_volume_velocity(&matrix, &rhs, &pressure, &cap_dofs, 2.0).unwrap();
        assert_complex_close(flow, Complex64::new(0.0, -1.5), 1e-14);
    }

    fn straight_config(mesh_size_m: f64) -> RoundPortLinearReferenceConfig {
        RoundPortLinearReferenceConfig {
            geometry: RoundPortGeometry {
                duct_length_m: 0.08,
                duct_radius_m: 0.003,
                chamber_length_m: 0.01,
                chamber_radius_m: 0.003,
                flare_radius_m: 0.0,
                mesh_size_m,
            },
            fluid: FluidProperties::air_20c(),
            box_volume_m3: 0.04,
        }
    }

    #[test]
    fn invalid_reference_inputs_return_typed_errors() {
        let config = straight_config(0.01);
        let mesh = super::super::build_round_port_mesh(&config.geometry).unwrap();
        for frequency in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            assert!(matches!(
                solve_round_port_linear_reference(&mesh, &config, frequency),
                Err(PortFlowError::LinearReference(_))
            ));
        }

        let wrong_dimension = Mesh::new(2);
        assert!(matches!(
            solve_round_port_linear_reference(&wrong_dimension, &config, 30.0),
            Err(PortFlowError::LinearReference(_))
        ));

        let mut missing_marker = mesh.clone();
        missing_marker
            .boundaries
            .retain(|boundary| boundary.marker != MARKER_EXTERIOR_CAP);
        assert!(matches!(
            solve_round_port_linear_reference(&missing_marker, &config, 30.0),
            Err(PortFlowError::LinearReference(_))
        ));

        let mut mismatched = config.clone();
        mismatched.geometry.duct_length_m *= 1.1;
        assert!(matches!(
            solve_round_port_linear_reference(&mesh, &mismatched, 30.0),
            Err(PortFlowError::LinearReference(_))
        ));

        let mut nonfinite_matrix = TripletMatrix::new(1, 1);
        nonfinite_matrix.push(0, 0, Complex64::new(1.0, 0.0));
        assert!(matches!(
            solve_pressure(&nonfinite_matrix.to_csr(), &[Complex64::new(f64::NAN, 0.0)]),
            Err(PortFlowError::LinearReference(_))
        ));

        assert!(matches!(
            checked_port_impedance(
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 0.0),
                &config.geometry,
                &config.fluid,
            ),
            Err(PortFlowError::LinearReference(_))
        ));
    }

    #[test]
    fn matching_mesh_validation_accepts_canonical_sharp_shoulders() {
        let mut config = straight_config(0.003);
        config.geometry.chamber_radius_m = 0.006;
        let mesh = super::super::build_round_port_mesh(&config.geometry).unwrap();
        validate_matching_mesh(&mesh, &config.geometry).unwrap();
    }

    #[test]
    fn matching_mesh_validation_rejects_corrupted_cap_markers() {
        let config = straight_config(0.01);
        let mesh = super::super::build_round_port_mesh(&config.geometry).unwrap();

        let mut swapped = mesh.clone();
        for boundary in &mut swapped.boundaries {
            boundary.marker = match boundary.marker {
                MARKER_BOX_CAP => MARKER_EXTERIOR_CAP,
                MARKER_EXTERIOR_CAP => MARKER_BOX_CAP,
                marker => marker,
            };
        }
        assert!(matches!(
            validate_matching_mesh(&swapped, &config.geometry),
            Err(PortFlowError::LinearReference(_))
        ));

        let mut partial = mesh.clone();
        let box_face = partial
            .boundaries
            .iter_mut()
            .find(|boundary| boundary.marker == MARKER_BOX_CAP)
            .unwrap();
        box_face.marker = MARKER_WALL;
        assert!(matches!(
            validate_matching_mesh(&partial, &config.geometry),
            Err(PortFlowError::LinearReference(_))
        ));

        let mut misplaced = mesh;
        let wall_face = misplaced
            .boundaries
            .iter_mut()
            .find(|boundary| boundary.marker == MARKER_WALL)
            .unwrap();
        wall_face.marker = MARKER_BOX_CAP;
        assert!(matches!(
            validate_matching_mesh(&misplaced, &config.geometry),
            Err(PortFlowError::LinearReference(_))
        ));
    }

    #[test]
    fn matching_mesh_validation_rejects_material_wall_radius_change() {
        let config = straight_config(0.01);
        let mut mesh = super::super::build_round_port_mesh(&config.geometry).unwrap();
        let node = mesh
            .boundaries
            .iter()
            .filter(|boundary| boundary.marker == MARKER_WALL)
            .flat_map(|boundary| boundary.nodes.iter().copied())
            .find(|&node| {
                let point = mesh.nodes[node];
                point.x > 0.0
                    && point.x < config.geometry.duct_length_m
                    && point.y.hypot(point.z) > 0.0
            })
            .expect("straight fixture should expose an interior wall node");
        mesh.nodes[node].y *= 1.05;
        mesh.nodes[node].z *= 1.05;
        assert!(matches!(
            validate_matching_mesh(&mesh, &config.geometry),
            Err(PortFlowError::LinearReference(_))
        ));
    }

    fn analytical_tube_impedance(
        config: &RoundPortLinearReferenceConfig,
        frequency_hz: f64,
    ) -> Complex64 {
        let omega = TAU * frequency_hz;
        let equivalent = EquivalentFluid::new(
            config.fluid.clone(),
            CrossSection::Circular {
                radius: config.geometry.duct_radius_m,
            },
        );
        let wave_number = equivalent.complex_wavenumber(omega);
        let volume_impedance = equivalent.complex_impedance(omega)
            / (std::f64::consts::PI * config.geometry.duct_radius_m.powi(2));
        let length = config.geometry.duct_length_m + 2.0 * config.geometry.chamber_length_m;
        // For exp(-i omega t), p=0 at x=L and box-to-exterior-positive flow
        // give Z_in = -i Zc tan(kL). This has Im(Z)<0 in the lossless limit.
        Complex64::new(0.0, -1.0) * volume_impedance * (wave_number * length).tan()
    }

    fn inviscid_tube_impedance(
        config: &RoundPortLinearReferenceConfig,
        frequency_hz: f64,
    ) -> Complex64 {
        let omega = TAU * frequency_hz;
        let bulk_modulus = config.fluid.ratio_of_specific_heats * config.fluid.ambient_pressure;
        let wave_number = omega * (config.fluid.density / bulk_modulus).sqrt();
        let area = std::f64::consts::PI * config.geometry.duct_radius_m.powi(2);
        let volume_impedance = (config.fluid.density * bulk_modulus).sqrt() / area;
        Complex64::new(0.0, -1.0)
            * volume_impedance
            * (wave_number
                * (config.geometry.duct_length_m + 2.0 * config.geometry.chamber_length_m))
                .tan()
    }

    #[test]
    fn wide_tube_limit_tends_to_inviscid_negative_imaginary_inertance() {
        let frequency = 30.0;
        let mut relative_errors = Vec::new();
        for radius in [0.003, 0.03, 1.0] {
            let mut config = straight_config(0.01);
            config.geometry.duct_radius_m = radius;
            config.geometry.chamber_radius_m = radius;
            let viscothermal = analytical_tube_impedance(&config, frequency);
            let inviscid = inviscid_tube_impedance(&config, frequency);
            assert!(viscothermal.re >= 0.0);
            assert!(viscothermal.im < 0.0);
            relative_errors.push((viscothermal - inviscid).norm() / inviscid.norm());
        }
        assert!(relative_errors.windows(2).all(|pair| pair[1] < pair[0]));
        assert!(
            relative_errors[2] < 1.0e-3,
            "wide-tube inviscid error={:e}",
            relative_errors[2]
        );
    }

    #[test]
    fn straight_tube_matches_zwikker_kosten_and_quadrature_is_converged() {
        let frequency = 30.0;
        // Keep the routine public-reference gate bounded now that the
        // authoritative solve is scalar P2. This short, wide tube is the
        // validated Stage-B analytical fixture (1,833 scalar P2 DOFs at the
        // present mesh generator settings), rather than the former P1-sized
        // 80 mm x 3 mm tube whose P2 promotion is prohibitively expensive.
        let config = RoundPortLinearReferenceConfig {
            geometry: RoundPortGeometry {
                duct_length_m: 0.02,
                duct_radius_m: 0.01,
                chamber_length_m: 0.005,
                chamber_radius_m: 0.01,
                flare_radius_m: 0.0,
                mesh_size_m: 0.006,
            },
            fluid: FluidProperties::air_20c(),
            box_volume_m3: 0.04,
        };
        let mesh = super::super::build_round_port_mesh(&config.geometry).unwrap();
        let numerical = solve_round_port_linear_reference(&mesh, &config, frequency).unwrap();
        let analytical = analytical_tube_impedance(&config, frequency);
        let error = (numerical.port_impedance_pa_s_per_m3 - analytical).norm() / analytical.norm();
        println!(
            "straight tube: Z_fem={:?}, Z_exact={analytical:?}, complex_error={error:e}, residual={:e}, strong_weak_flux_delta={:e}, weak_net_storage={:?}, H_condition={:e}",
            numerical.port_impedance_pa_s_per_m3,
            numerical.relative_residual,
            numerical.strong_weak_exterior_flux_relative_difference,
            numerical.weak_net_storage_volume_velocity_rms_m3_per_s,
            numerical.box_to_port_flow_ratio_condition_number,
        );
        assert!(
            error < 0.02,
            "complex tube error={error:e}, numerical={:?}, analytical={analytical:?}",
            numerical.port_impedance_pa_s_per_m3
        );
        assert!(numerical.port_impedance_pa_s_per_m3.re >= 0.0);
        assert!(numerical.port_impedance_pa_s_per_m3.im < 0.0);
        assert!(
            numerical
                .strong_weak_exterior_flux_relative_difference
                .is_finite()
        );
        assert!(
            numerical
                .box_to_port_flow_ratio_condition_number
                .is_finite()
        );

        assert_eq!(numerical.polynomial_degree, 2);
        assert_eq!(numerical.quadrature_order, 5);
        assert_eq!(
            numerical.scalar_dofs,
            H1Space::new(PolynomialDegree::P2).num_global_dofs(&mesh)
        );
        let lower = solve_with_degree_and_quadrature_order(
            &mesh,
            &config,
            frequency,
            PolynomialDegree::P2,
            4,
        )
        .unwrap();
        let quadrature_delta =
            (lower.port_impedance_pa_s_per_m3 - numerical.port_impedance_pa_s_per_m3).norm()
                / numerical.port_impedance_pa_s_per_m3.norm();
        assert!(
            quadrature_delta < 0.005,
            "quadrature delta={quadrature_delta:e}"
        );

        let compliance =
            config.box_volume_m3 / (config.fluid.density * config.fluid.speed_of_sound.powi(2));
        let expected_h = Complex64::new(1.0, 0.0)
            / (Complex64::new(1.0, 0.0)
                - Complex64::new(0.0, TAU * frequency * compliance)
                    * numerical.port_impedance_pa_s_per_m3);
        assert_eq!(numerical.box_to_port_flow_ratio, expected_h);
    }

    #[test]
    fn p1_straight_tube_refinement_improves_complex_impedance() {
        let frequency_hz = 30.0;
        let coarse_config = straight_config(0.006);
        let fine_config = straight_config(0.004);
        let coarse_mesh = super::super::build_round_port_mesh(&coarse_config.geometry).unwrap();
        let fine_mesh = super::super::build_round_port_mesh(&fine_config.geometry).unwrap();
        let exact = analytical_tube_impedance(&fine_config, frequency_hz);

        let coarse = solve_p1_with_quadrature_order(
            &coarse_mesh,
            &coarse_config,
            frequency_hz,
            STAGE_B_P2_QUADRATURE_ORDER,
        )
        .unwrap();
        let fine = solve_p1_with_quadrature_order(
            &fine_mesh,
            &fine_config,
            frequency_hz,
            STAGE_B_P2_QUADRATURE_ORDER,
        )
        .unwrap();
        let coarse_error = (coarse.port_impedance_pa_s_per_m3 - exact).norm() / exact.norm();
        let fine_error = (fine.port_impedance_pa_s_per_m3 - exact).norm() / exact.norm();

        assert!(
            fine_error < coarse_error,
            "P1 straight-tube refinement did not improve the analytic impedance: coarse={coarse_error:e}, fine={fine_error:e}"
        );
    }

    #[test]
    fn bounded_flared_reference_changes_by_less_than_one_percent() {
        let geometry = super::super::geometry3d::RoundPortGeometry {
            duct_length_m: 0.02,
            duct_radius_m: 0.01,
            chamber_length_m: 0.005,
            chamber_radius_m: 0.015,
            flare_radius_m: 0.002,
            mesh_size_m: 0.02,
        };
        let config = RoundPortLinearReferenceConfig {
            geometry,
            fluid: FluidProperties::air_20c(),
            box_volume_m3: 0.005,
        };
        let frequency_hz = config.fluid.speed_of_sound
            / (TAU
                * (config.box_volume_m3
                    / (std::f64::consts::PI
                        * geometry.duct_radius_m.powi(2)
                        * geometry.duct_length_m))
                    .sqrt());
        let coarse = super::super::build_round_port_mesh(&geometry).unwrap();
        let mut fine = coarse.clone();
        super::super::refine_round_port_mesh_projected(&mut fine, &geometry).unwrap();

        let coarse_result = solve_with_degree_and_quadrature_order(
            &coarse,
            &config,
            frequency_hz,
            PolynomialDegree::P2,
            STAGE_B_P2_QUADRATURE_ORDER,
        )
        .unwrap();
        let fine_result = solve_with_degree_and_quadrature_order(
            &fine,
            &config,
            frequency_hz,
            PolynomialDegree::P2,
            STAGE_B_P2_QUADRATURE_ORDER,
        )
        .unwrap();
        let relative_change = |coarse: Complex64, fine: Complex64| {
            (coarse - fine).norm() / fine.norm().max(f64::EPSILON)
        };
        let impedance_change = relative_change(
            coarse_result.port_impedance_pa_s_per_m3,
            fine_result.port_impedance_pa_s_per_m3,
        );
        let transfer_change = relative_change(
            coarse_result.box_to_port_flow_ratio,
            fine_result.box_to_port_flow_ratio,
        );
        assert!(
            impedance_change < 0.01,
            "bounded flared impedance refinement change={impedance_change:e}"
        );
        assert!(
            transfer_change < 0.01,
            "bounded flared transfer refinement change={transfer_change:e}"
        );
        assert!(coarse_result.relative_residual < REFERENCE_RESIDUAL_TOLERANCE);
        assert!(fine_result.relative_residual < REFERENCE_RESIDUAL_TOLERANCE);
    }

    #[test]
    #[ignore = "release-mode straight-tube h-refinement evidence; the routine P2 analytical gate is bounded above"]
    fn straight_tube_refinement_improves_complex_impedance() {
        let frequency = 30.0;
        let coarse = straight_config(0.0015);
        let fine = straight_config(0.00075);
        let coarse_result = solve_round_port_linear_reference(
            &super::super::build_round_port_mesh(&coarse.geometry).unwrap(),
            &coarse,
            frequency,
        )
        .unwrap();
        let fine_result = solve_round_port_linear_reference(
            &super::super::build_round_port_mesh(&fine.geometry).unwrap(),
            &fine,
            frequency,
        )
        .unwrap();
        let exact = analytical_tube_impedance(&fine, frequency);
        let coarse_error = (coarse_result.port_impedance_pa_s_per_m3 - exact).norm() / exact.norm();
        let fine_error = (fine_result.port_impedance_pa_s_per_m3 - exact).norm() / exact.norm();
        println!("straight tube refinement: coarse={coarse_error:e}, fine={fine_error:e}");
        assert!(
            fine_error < coarse_error,
            "fine={fine_error:e}, coarse={coarse_error:e}"
        );
    }

    #[test]
    fn p2_straight_tube_matches_analytical_and_improves_with_h_refinement() {
        let frequency_hz = 30.0;
        let make_config = |mesh_size_m| RoundPortLinearReferenceConfig {
            geometry: RoundPortGeometry {
                duct_length_m: 0.02,
                duct_radius_m: 0.01,
                chamber_length_m: 0.005,
                chamber_radius_m: 0.01,
                flare_radius_m: 0.0,
                mesh_size_m,
            },
            fluid: FluidProperties::air_20c(),
            box_volume_m3: 0.04,
        };
        // The fine mesh also refines the circular polygon from the generator's
        // 20-sector minimum to 32 sectors, so the analytical comparison is not
        // artificially frozen by identical boundary geometry.
        let coarse_config = make_config(0.006);
        let fine_config = make_config(0.002);
        let coarse_mesh = super::super::build_round_port_mesh(&coarse_config.geometry).unwrap();
        let fine_mesh = super::super::build_round_port_mesh(&fine_config.geometry).unwrap();
        let exact = analytical_tube_impedance(&coarse_config, frequency_hz);

        let p1_coarse = solve_p1_with_quadrature_order(
            &coarse_mesh,
            &coarse_config,
            frequency_hz,
            STAGE_B_P2_QUADRATURE_ORDER,
        )
        .unwrap();
        let p2_coarse = solve_with_degree_and_quadrature_order(
            &coarse_mesh,
            &coarse_config,
            frequency_hz,
            PolynomialDegree::P2,
            STAGE_B_P2_QUADRATURE_ORDER,
        )
        .unwrap();
        let p2_fine = solve_with_degree_and_quadrature_order(
            &fine_mesh,
            &fine_config,
            frequency_hz,
            PolynomialDegree::P2,
            STAGE_B_P2_QUADRATURE_ORDER,
        )
        .unwrap();
        let relative_error = |value: Complex64| (value - exact).norm() / exact.norm();
        let p1_coarse_error = relative_error(p1_coarse.port_impedance_pa_s_per_m3);
        let p2_coarse_error = relative_error(p2_coarse.port_impedance_pa_s_per_m3);
        let p2_fine_error = relative_error(p2_fine.port_impedance_pa_s_per_m3);
        let space = H1Space::new(PolynomialDegree::P2);
        println!(
            "P2_STRAIGHT_ANALYTICAL coarse_nodes={} coarse_tetrahedra={} coarse_p2_dofs={} fine_nodes={} fine_tetrahedra={} fine_p2_dofs={} z_exact={exact:?} z_p1_coarse={:?} z_p2_coarse={:?} z_p2_fine={:?} p1_coarse_error={p1_coarse_error:.12e} p2_coarse_error={p2_coarse_error:.12e} p2_fine_error={p2_fine_error:.12e}",
            coarse_mesh.num_nodes(),
            coarse_mesh.elements.len(),
            space.num_global_dofs(&coarse_mesh),
            fine_mesh.num_nodes(),
            fine_mesh.elements.len(),
            space.num_global_dofs(&fine_mesh),
            p1_coarse.port_impedance_pa_s_per_m3,
            p2_coarse.port_impedance_pa_s_per_m3,
            p2_fine.port_impedance_pa_s_per_m3,
        );
        assert!(
            p2_coarse_error <= p1_coarse_error,
            "P2 coarse error {p2_coarse_error:e} exceeds P1 {p1_coarse_error:e}"
        );
        assert!(
            p2_fine_error < p2_coarse_error,
            "P2 h refinement did not improve: coarse={p2_coarse_error:e}, fine={p2_fine_error:e}"
        );
        assert!(
            p2_fine_error < 0.01,
            "P2 fine analytical complex error {p2_fine_error:e} exceeds 1%"
        );
    }

    #[test]
    #[ignore = "release-mode Stage-B flared P2 quadrature evidence"]
    fn p2_flared_order_four_and_degree_five_quadrature_agree() {
        let geometry = RoundPortGeometry {
            duct_length_m: 0.15,
            duct_radius_m: 0.025,
            chamber_length_m: 0.05,
            chamber_radius_m: 0.08,
            flare_radius_m: 0.01,
            mesh_size_m: 0.04,
        };
        let config = RoundPortLinearReferenceConfig {
            geometry,
            fluid: FluidProperties {
                density: 1.2,
                speed_of_sound: 343.0,
                dynamic_viscosity: 1.85e-5,
                ..FluidProperties::air_20c()
            },
            box_volume_m3: 0.005,
        };
        let area = std::f64::consts::PI * geometry.duct_radius_m.powi(2);
        let frequency_hz = config.fluid.speed_of_sound / TAU
            * (area / (config.box_volume_m3 * geometry.duct_length_m)).sqrt();
        let mesh = super::super::build_round_port_mesh(&geometry).unwrap();
        let order_four = solve_with_degree_and_quadrature_order(
            &mesh,
            &config,
            frequency_hz,
            PolynomialDegree::P2,
            4,
        )
        .unwrap();
        let degree_five = solve_with_degree_and_quadrature_order(
            &mesh,
            &config,
            frequency_hz,
            PolynomialDegree::P2,
            STAGE_B_P2_QUADRATURE_ORDER,
        )
        .unwrap();
        let z_delta =
            (order_four.port_impedance_pa_s_per_m3 - degree_five.port_impedance_pa_s_per_m3).norm()
                / degree_five.port_impedance_pa_s_per_m3.norm();
        println!(
            "P2_FLARED_QUADRATURE nodes={} tetrahedra={} scalar_p2_dofs={} frequency_hz={frequency_hz:.9} z_order4={:?} z_degree5={:?} z_delta={z_delta:.12e} residual_order4={:.12e} residual_degree5={:.12e}",
            mesh.num_nodes(),
            mesh.elements.len(),
            H1Space::new(PolynomialDegree::P2).num_global_dofs(&mesh),
            order_four.port_impedance_pa_s_per_m3,
            degree_five.port_impedance_pa_s_per_m3,
            order_four.relative_residual,
            degree_five.relative_residual,
        );
        assert!(
            z_delta < 0.005,
            "full-complex flared P2 order-4/degree-5 Z delta {z_delta:e} exceeds 0.5%"
        );
    }

    #[test]
    #[ignore = "release-mode Stage-B P2 projected nested convergence probe"]
    fn p2_topology_nested_geometry_projected_convergence_probe() {
        let geometry = RoundPortGeometry {
            duct_length_m: 0.15,
            duct_radius_m: 0.025,
            chamber_length_m: 0.05,
            chamber_radius_m: 0.08,
            flare_radius_m: 0.01,
            mesh_size_m: 0.02,
        };
        let config = RoundPortLinearReferenceConfig {
            geometry,
            fluid: FluidProperties {
                density: 1.2,
                speed_of_sound: 343.0,
                dynamic_viscosity: 1.85e-5,
                ..FluidProperties::air_20c()
            },
            box_volume_m3: 0.005,
        };
        let area = std::f64::consts::PI * geometry.duct_radius_m.powi(2);
        let frequency_hz = config.fluid.speed_of_sound / TAU
            * (area / (config.box_volume_m3 * geometry.duct_length_m)).sqrt();
        let coarse = super::super::build_round_port_mesh(&geometry).unwrap();
        let mut fine = coarse.clone();
        super::super::refine_round_port_mesh_projected(&mut fine, &geometry).unwrap();
        let space = H1Space::new(PolynomialDegree::P2);
        println!(
            "P2_PROBE_DOMAIN duct_length_m={} duct_radius_m={} chamber_length_m={} chamber_radius_m={} flare_radius_m={} base_mesh_size_m={} frequency_hz={:.9}",
            geometry.duct_length_m,
            geometry.duct_radius_m,
            geometry.chamber_length_m,
            geometry.chamber_radius_m,
            geometry.flare_radius_m,
            geometry.mesh_size_m,
            frequency_hz,
        );
        println!(
            "P2_PROBE_MESH coarse_nodes={} coarse_tetrahedra={} coarse_scalar_dofs={} fine_nodes={} fine_tetrahedra={} fine_scalar_dofs={}",
            coarse.num_nodes(),
            coarse.elements.len(),
            space.num_global_dofs(&coarse),
            fine.num_nodes(),
            fine.elements.len(),
            space.num_global_dofs(&fine),
        );

        let coarse_started = std::time::Instant::now();
        let coarse_result = solve_with_degree_and_quadrature_order(
            &coarse,
            &config,
            frequency_hz,
            PolynomialDegree::P2,
            STAGE_B_P2_QUADRATURE_ORDER,
        )
        .unwrap();
        let coarse_elapsed_s = coarse_started.elapsed().as_secs_f64();
        let fine_started = std::time::Instant::now();
        let fine_result = solve_with_degree_and_quadrature_order(
            &fine,
            &config,
            frequency_hz,
            PolynomialDegree::P2,
            STAGE_B_P2_QUADRATURE_ORDER,
        )
        .unwrap();
        let fine_elapsed_s = fine_started.elapsed().as_secs_f64();
        let z_delta = (coarse_result.port_impedance_pa_s_per_m3
            - fine_result.port_impedance_pa_s_per_m3)
            .norm()
            / fine_result.port_impedance_pa_s_per_m3.norm();
        let h_delta = (coarse_result.box_to_port_flow_ratio - fine_result.box_to_port_flow_ratio)
            .norm()
            / fine_result.box_to_port_flow_ratio.norm();
        println!(
            "P2_PROBE_RESULT z_coarse={:?} z_fine={:?} h_coarse={:?} h_fine={:?} z_delta={:.12e} h_delta={:.12e} coarse_residual={:.12e} fine_residual={:.12e} coarse_strong_weak={:.12e} fine_strong_weak={:.12e} coarse_elapsed_s={:.6} fine_elapsed_s={:.6}",
            coarse_result.port_impedance_pa_s_per_m3,
            fine_result.port_impedance_pa_s_per_m3,
            coarse_result.box_to_port_flow_ratio,
            fine_result.box_to_port_flow_ratio,
            z_delta,
            h_delta,
            coarse_result.relative_residual,
            fine_result.relative_residual,
            coarse_result.strong_weak_exterior_flux_relative_difference,
            fine_result.strong_weak_exterior_flux_relative_difference,
            coarse_elapsed_s,
            fine_elapsed_s,
        );
        assert!(
            z_delta < 0.01,
            "full-complex P2 topology-nested, geometry-projected Z delta {z_delta:e} exceeds 1%"
        );
        assert!(
            h_delta < 0.01,
            "full-complex P2 topology-nested, geometry-projected H delta {h_delta:e} exceeds 1%"
        );
    }

    // ---- PORT-OPTIM Step 4: Tier 1 spline linear reference ----

    use super::super::duct1d::{Duct1dConfig, solve_duct_1d, tuning_frequency_hz};
    use super::super::profile::{ProfileOptions, SplinePortProfile};
    use crate::AcousticMedium;

    fn spline_reference_geometry(
        duct_length_m: f64,
        bore_radius_m: f64,
        chamber_length_m: f64,
        chamber_radius_m: f64,
        flare_length_m: f64,
        mesh_size_m: f64,
        profile: SplinePortProfile,
    ) -> SplineRoundPortGeometry {
        SplineRoundPortGeometry {
            duct_length_m,
            bore_radius_m,
            chamber_length_m,
            chamber_radius_m,
            flare_length_m,
            mesh_size_m,
            angular_sectors: None,
            profile,
        }
    }

    fn relative_complex_difference(actual: Complex64, expected: Complex64) -> f64 {
        (actual - expected).norm() / expected.norm().max(f64::EPSILON)
    }

    #[test]
    fn spline_straight_profile_reproduces_smoothstep_flare_reference() {
        // Same bounded flared fixture as the refinement gate above; the spline
        // path carries a straight profile with identical chambers and flare,
        // so its radius function is the smoothstep reference up to
        // spline-evaluation roundoff.
        let geometry = RoundPortGeometry {
            duct_length_m: 0.02,
            duct_radius_m: 0.01,
            chamber_length_m: 0.005,
            chamber_radius_m: 0.015,
            flare_radius_m: 0.002,
            mesh_size_m: 0.02,
        };
        let fluid = FluidProperties::air_20c();
        let round_config = RoundPortLinearReferenceConfig {
            geometry,
            fluid: fluid.clone(),
            box_volume_m3: 0.005,
        };
        let profile = SplinePortProfile::straight(
            5,
            geometry.duct_length_m,
            geometry.duct_radius_m,
            0.6 * geometry.duct_radius_m,
            ProfileOptions::default(),
        )
        .expect("straight test profile must construct");
        let spline_geometry = spline_reference_geometry(
            geometry.duct_length_m,
            geometry.duct_radius_m,
            geometry.chamber_length_m,
            geometry.chamber_radius_m,
            geometry.flare_radius_m,
            geometry.mesh_size_m,
            profile,
        );
        let spline_config = SplinePortLinearReferenceConfig {
            geometry: spline_geometry.clone(),
            fluid,
            box_volume_m3: 0.005,
        };
        let round_mesh = super::super::build_round_port_mesh(&geometry).unwrap();
        let spline_mesh = super::super::build_spline_round_port_mesh(&spline_geometry).unwrap();
        assert_eq!(spline_mesh.num_nodes(), round_mesh.num_nodes());
        assert_eq!(spline_mesh.elements.len(), round_mesh.elements.len());
        assert_eq!(spline_mesh.boundaries.len(), round_mesh.boundaries.len());
        let max_coordinate_difference = spline_mesh
            .nodes
            .iter()
            .zip(round_mesh.nodes.iter())
            .map(|(a, b)| {
                (a.x - b.x)
                    .abs()
                    .max((a.y - b.y).abs())
                    .max((a.z - b.z).abs())
            })
            .fold(0.0f64, f64::max);
        assert!(
            max_coordinate_difference < 1.0e-12,
            "spline/round mesh coordinate difference={max_coordinate_difference:e}"
        );

        let frequency_hz = 90.0;
        let round =
            solve_round_port_linear_reference(&round_mesh, &round_config, frequency_hz).unwrap();
        let spline =
            solve_spline_port_linear_reference(&spline_mesh, &spline_config, frequency_hz).unwrap();
        let impedance_difference = relative_complex_difference(
            spline.port_impedance_pa_s_per_m3,
            round.port_impedance_pa_s_per_m3,
        );
        let transfer_difference = relative_complex_difference(
            spline.box_to_port_flow_ratio,
            round.box_to_port_flow_ratio,
        );
        let flow_difference = relative_complex_difference(
            spline.port_volume_velocity_rms_m3_per_s,
            round.port_volume_velocity_rms_m3_per_s,
        );
        println!(
            "spline==smoothstep: Z_round={:?} Z_spline={:?} z_delta={impedance_difference:e} h_delta={transfer_difference:e} q_delta={flow_difference:e} residual={:e} strong_weak={:e}",
            round.port_impedance_pa_s_per_m3,
            spline.port_impedance_pa_s_per_m3,
            spline.relative_residual,
            spline.strong_weak_exterior_flux_relative_difference,
        );
        assert!(
            impedance_difference < 1.0e-9,
            "spline/round impedance difference={impedance_difference:e}"
        );
        assert!(
            transfer_difference < 1.0e-9,
            "spline/round transfer difference={transfer_difference:e}"
        );
        assert!(
            flow_difference < 1.0e-9,
            "spline/round flow difference={flow_difference:e}"
        );
        assert!(spline.relative_residual <= REFERENCE_RESIDUAL_TOLERANCE);
        assert!(
            spline
                .strong_weak_exterior_flux_relative_difference
                .is_finite()
        );
    }

    #[test]
    fn spline_straight_tube_matches_round_tube_and_zwikker_kosten() {
        // Straight-cylinder limit (chambers match the bore, zero flare) on the
        // validated Stage-B analytical fixture.
        let geometry = RoundPortGeometry {
            duct_length_m: 0.02,
            duct_radius_m: 0.01,
            chamber_length_m: 0.005,
            chamber_radius_m: 0.01,
            flare_radius_m: 0.0,
            mesh_size_m: 0.006,
        };
        let frequency_hz = 30.0;
        let round_config = RoundPortLinearReferenceConfig {
            geometry,
            fluid: FluidProperties::air_20c(),
            box_volume_m3: 0.04,
        };
        let profile = SplinePortProfile::straight(
            5,
            geometry.duct_length_m,
            geometry.duct_radius_m,
            0.6 * geometry.duct_radius_m,
            ProfileOptions::default(),
        )
        .expect("straight test profile must construct");
        let spline_geometry = spline_reference_geometry(
            geometry.duct_length_m,
            geometry.duct_radius_m,
            geometry.chamber_length_m,
            geometry.chamber_radius_m,
            0.0,
            geometry.mesh_size_m,
            profile,
        );
        let spline_config = SplinePortLinearReferenceConfig {
            geometry: spline_geometry.clone(),
            fluid: FluidProperties::air_20c(),
            box_volume_m3: 0.04,
        };
        let round_mesh = super::super::build_round_port_mesh(&geometry).unwrap();
        let spline_mesh = super::super::build_spline_round_port_mesh(&spline_geometry).unwrap();
        assert_eq!(spline_mesh.num_nodes(), round_mesh.num_nodes());
        assert_eq!(spline_mesh.elements.len(), round_mesh.elements.len());

        let round =
            solve_round_port_linear_reference(&round_mesh, &round_config, frequency_hz).unwrap();
        let spline =
            solve_spline_port_linear_reference(&spline_mesh, &spline_config, frequency_hz).unwrap();
        let analytical = analytical_tube_impedance(&round_config, frequency_hz);
        let path_difference = relative_complex_difference(
            spline.port_impedance_pa_s_per_m3,
            round.port_impedance_pa_s_per_m3,
        );
        let analytical_error =
            relative_complex_difference(spline.port_impedance_pa_s_per_m3, analytical);
        println!(
            "spline straight tube: Z_spline={:?} Z_round={:?} Z_exact={analytical:?} path_delta={path_difference:e} analytical_error={analytical_error:e} residual={:e}",
            spline.port_impedance_pa_s_per_m3,
            round.port_impedance_pa_s_per_m3,
            spline.relative_residual,
        );
        assert!(
            path_difference < 1.0e-9,
            "spline/round straight-tube difference={path_difference:e}"
        );
        assert!(
            analytical_error < 0.02,
            "spline straight-tube analytical error={analytical_error:e}"
        );
        assert!(spline.port_impedance_pa_s_per_m3.re >= 0.0);
        assert!(spline.port_impedance_pa_s_per_m3.im < 0.0);
        assert!(spline.relative_residual <= REFERENCE_RESIDUAL_TOLERANCE);
    }

    #[test]
    fn spline_mesh_validation_rejects_wrong_profile() {
        let straight = SplinePortProfile::straight(5, 0.02, 0.01, 0.006, ProfileOptions::default())
            .expect("straight test profile must construct");
        let dipped = SplinePortProfile::from_normalized(
            5,
            0.02,
            0.01,
            0.006,
            &[0.3, 0.3],
            ProfileOptions::default(),
        )
        .expect("dipped test profile must construct");
        let straight_geometry =
            spline_reference_geometry(0.02, 0.01, 0.005, 0.01, 0.0, 0.008, straight);
        let dipped_geometry =
            spline_reference_geometry(0.02, 0.01, 0.005, 0.01, 0.0, 0.008, dipped);
        let straight_mesh = super::super::build_spline_round_port_mesh(&straight_geometry).unwrap();
        let dipped_mesh = super::super::build_spline_round_port_mesh(&dipped_geometry).unwrap();
        validate_matching_mesh(&straight_mesh, &straight_geometry).unwrap();
        validate_matching_mesh(&dipped_mesh, &dipped_geometry).unwrap();
        // Cross-validation must fail in both directions: the straight mesh
        // exceeds the dipped wall, and the dipped mesh undershoots the
        // straight wall beyond the chord tolerance.
        assert!(matches!(
            validate_matching_mesh(&straight_mesh, &dipped_geometry),
            Err(PortFlowError::LinearReference(_))
        ));
        assert!(matches!(
            validate_matching_mesh(&dipped_mesh, &straight_geometry),
            Err(PortFlowError::LinearReference(_))
        ));
    }

    #[test]
    fn tier0_tier1_tuning_agrees_within_1hz_on_test_profiles() {
        const DUCT_LENGTH_M: f64 = 0.04;
        const BORE_RADIUS_M: f64 = 0.025;
        const THROAT_RADIUS_M: f64 = 0.015;
        const BOX_VOLUME_M3: f64 = 0.02;
        // Straight-cylinder Tier 1 with matched total acoustic length: the
        // chambers stand in for the Tier 0 end regions, so each chamber
        // carries half of the Tier 0 interior + radiation end correction
        // (area-ratio adjusted). Both tiers then reduce to the same lumped
        // circuit up to distributed, viscothermal, and discretization effects.
        // The wide bore keeps the viscothermal slowing (δ/r ~ 1%) and the
        // explicit 48 sectors keep the polygon-area deficit (~0.1%) small, so
        // the residual Tier 0 (lossless compact) vs Tier 1 (resolved
        // viscothermal) tuning offset stays well inside the 1 Hz gate.
        let medium = AcousticMedium {
            density_kg_per_m3: 1.2,
            sound_speed_m_per_s: 343.0,
            dynamic_viscosity_pa_s: 1.85e-5,
            thermal_diffusivity_m2_per_s: 2.2e-5,
        };
        let fluid = FluidProperties {
            density: 1.2,
            speed_of_sound: 343.0,
            dynamic_viscosity: 1.85e-5,
            ..FluidProperties::air_20c()
        };
        let duct_config = Duct1dConfig {
            n_segments: super::super::duct1d::DUCT1D_DEFAULT_SEGMENTS,
            thermoviscous: false,
            box_volume_m3: BOX_VOLUME_M3,
        };
        let straight = SplinePortProfile::straight(
            5,
            DUCT_LENGTH_M,
            BORE_RADIUS_M,
            THROAT_RADIUS_M,
            ProfileOptions::default(),
        )
        .expect("straight test profile must construct");
        let dipped = SplinePortProfile::from_normalized(
            5,
            DUCT_LENGTH_M,
            BORE_RADIUS_M,
            THROAT_RADIUS_M,
            &[0.3, 0.3],
            ProfileOptions::default(),
        )
        .expect("dipped test profile must construct");
        for (name, profile) in [("straight", straight), ("dipped", dipped)] {
            let tuning_tier0 = tuning_frequency_hz(&profile, medium, &duct_config).unwrap();
            let tier0_point = solve_duct_1d(&profile, medium, tuning_tier0, &duct_config).unwrap();
            let chamber_area = std::f64::consts::PI * BORE_RADIUS_M * BORE_RADIUS_M;
            let interior_area = profile.area_m2(0.0);
            let exterior_area = std::f64::consts::PI
                * tier0_point.effective_exterior_radius_m
                * tier0_point.effective_exterior_radius_m;
            let chamber_length_m = 0.5
                * (tier0_point.interior_end_correction_m * chamber_area / interior_area
                    + tier0_point.exterior_end_correction_m * chamber_area / exterior_area);
            let mut spline_geometry = spline_reference_geometry(
                DUCT_LENGTH_M,
                BORE_RADIUS_M,
                chamber_length_m,
                BORE_RADIUS_M,
                0.0,
                0.01,
                profile,
            );
            spline_geometry.angular_sectors = Some(48);
            let spline_config = SplinePortLinearReferenceConfig {
                geometry: spline_geometry.clone(),
                fluid: fluid.clone(),
                box_volume_m3: BOX_VOLUME_M3,
            };
            let mesh = super::super::build_spline_round_port_mesh(&spline_geometry).unwrap();
            // Tier 1 tuning is the box-to-port transfer peak. Scan ±1.5 Hz at
            // 0.5 Hz around the Tier 0 prediction (the lightly damped peak is
            // ~1 Hz wide), then refine parabolically.
            let mut magnitudes = Vec::new();
            for step in -3..=3 {
                let frequency = tuning_tier0 + 0.5 * step as f64;
                let transfer =
                    solve_spline_port_linear_reference(&mesh, &spline_config, frequency).unwrap();
                assert!(transfer.relative_residual <= REFERENCE_RESIDUAL_TOLERANCE);
                assert!(
                    transfer
                        .strong_weak_exterior_flux_relative_difference
                        .is_finite()
                );
                magnitudes.push((frequency, transfer.box_to_port_flow_ratio.norm()));
            }
            let peak = magnitudes
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.1.total_cmp(&b.1))
                .map(|(index, _)| index)
                .expect("non-empty scan has a peak");
            assert!(
                (1..magnitudes.len() - 1).contains(&peak),
                "{name}: |H| peak at scan edge, cannot refine"
            );
            let (y0, y1, y2) = (
                magnitudes[peak - 1].1,
                magnitudes[peak].1,
                magnitudes[peak + 1].1,
            );
            let denominator = y0 - 2.0 * y1 + y2;
            let tuning_tier1 = if denominator.abs() > f64::EPSILON {
                // 0.5 Hz scan step: parabolic offset is (h/2)·(y0−y2)/denom.
                magnitudes[peak].0 + 0.25 * (y0 - y2) / denominator
            } else {
                magnitudes[peak].0
            };
            let difference_hz = (tuning_tier1 - tuning_tier0).abs();
            println!(
                "tier0/tier1 tuning {name}: f0_tier0={tuning_tier0:.6} f0_tier1={tuning_tier1:.6} delta={difference_hz:.6} chamber_length={chamber_length_m:.6e} cutoff_margin={:.3}",
                tier0_point.plane_wave_margin,
            );
            assert!(
                difference_hz <= 1.0,
                "{name}: Tier 0/Tier 1 tuning difference {difference_hz:.4} Hz exceeds 1 Hz"
            );
        }
    }
}
