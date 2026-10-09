//! Cached constant matrices for BDF Navier–Stokes time stepping.
//!
//! In a BDF time loop only the Oseen convection `C(u*)` changes from step to
//! step; the mass, viscous, and divergence blocks depend only on the mesh and
//! function spaces. [`BdfWorkspace`] assembles those four blocks once so that
//! [`super::navier_stokes::solve_bdf_step_cached`] can rebuild the monolithic
//! system per step as `mass_coeff·M + mu_step·K_unit + C(u*) + div blocks`
//! without re-running element assembly.
//!
//! The viscous block is stored with **unit viscosity**
//! (`StokesViscousForm::new(1.0)`): the per-step scaling is `mu_step * K_unit`,
//! where `mu_step` is `config.mu` today and becomes the effective viscosity
//! (laminar + turbulent) in a later step of the port-flow work.

use crate::assembly::TripletMatrix;
use crate::mesh::ElementType;
use std::cell::RefCell;

/// Linear solver used by the most recent cached BDF step.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BdfLinearSolverKind {
    /// Sparse direct LU.
    Direct,
    /// Monolithic GMRES with ILU.
    GmresIlu,
    /// FGMRES with an upper-triangular Jacobi/pressure-mass Schur
    /// preconditioner.
    FgmresJacobiSchur,
    /// FGMRES with upper-triangular ILUT velocity/pressure-mass Schur blocks.
    FgmresIlutSchur,
    /// Backend-aware FGMRES path executed through CUDA or Metal.
    BackendFgmres,
}

/// Diagnostics for the most recent cached BDF linear solve.
#[derive(Debug, Clone, Copy)]
pub struct BdfLinearSolveTelemetry {
    /// Selected solver path.
    pub solver: BdfLinearSolverKind,
    /// Concrete execution backend used by the cached BDF solve.
    ///
    /// The backend actually used by the cached BDF solve. Unavailable or
    /// failed accelerator setup is reported as `Cpu` after checked fallback.
    pub resolved_linear_solver_backend: super::navier_stokes::LinearSolverBackend,
    /// Krylov matrix-vector products, or zero for a direct solve.
    pub iterations: usize,
    /// Independently recomputed `||b-Ax||₂ / max(||b||₂, min-positive)`.
    pub true_relative_residual: f64,
    /// Whether an exactly matching factorization or preconditioner was reused.
    pub setup_reused: bool,
    /// Whether the solve met its configured acceptance contract.
    pub converged: bool,
}

#[allow(clippy::large_enum_variant)]
pub(crate) enum BdfCachedLinearSolver {
    Direct {
        matrix: sonium_solvers::CsrMatrix<f64>,
        factorization: sonium_solvers::direct::SparseLuFactorization<f64>,
    },
    JacobiSchur {
        matrix: sonium_solvers::CsrMatrix<f64>,
        preconditioner: sonium_solvers::SaddlePointPreconditioner<f64>,
    },
    Ilu {
        matrix: sonium_solvers::CsrMatrix<f64>,
        preconditioner: sonium_solvers::IluPreconditioner<f64>,
    },
}

/// Constant per-step matrices for the BDF Navier–Stokes saddle system.
pub struct BdfWorkspace {
    /// Velocity mass matrix `M` (unscaled; BDF coefficient applied per step).
    pub mass_triplet: TripletMatrix<f64>,
    /// Stokes viscous stiffness assembled with μ = 1; scaled by `mu_step`
    /// when the monolithic system is built.
    pub viscous_unit_triplet: TripletMatrix<f64>,
    /// Velocity–pressure coupling block (divergence/gradient), `(0,1)` block.
    pub block_01: TripletMatrix<f64>,
    /// Pressure–velocity coupling block, `(1,0)` block.
    pub block_10: TripletMatrix<f64>,
    /// Number of velocity DOFs.
    pub n_v: usize,
    /// Number of pressure DOFs.
    pub n_p: usize,
    /// Pressure DOF nearest the mesh centroid, used to fix the incompressible
    /// gauge consistently in steady and transient saddle systems.
    pub pressure_pin: usize,
    /// Prescribed pressure at [`Self::pressure_pin`]. The default centroid
    /// gauge is zero; traction-driven applications may instead pin a boundary
    /// DOF to the pressure prescribed on that boundary.
    pub pressure_pin_value: f64,
    /// Physical coordinate of every velocity DOF. Components of the same
    /// scalar support point share one coordinate (`dof = scalar * comp + c`);
    /// for P1 these are exactly the mesh node coordinates.
    pub velocity_dof_coordinates: Vec<[f64; 3]>,
    /// Mesh/space-invariant element lookup used to evaluate the extrapolated
    /// finite-element velocity exactly at convection quadrature points.
    pub(crate) element_velocity_topology: ElementVelocityTopology,
    /// Quadrature order used for the velocity-space assemblies: 2 for P1
    /// (legacy behaviour) and 5 for P2. Although P2 mass needs only degree 4,
    /// the Oseen integrand `w(P2) · grad(u)(P1) · v(P2)` has degree 5.
    pub quad_order: usize,
    /// Reusable setup for consecutive solves with an exactly identical
    /// monolithic matrix (notably the two affine pressure probes).
    pub(crate) linear_solver_cache: RefCell<Option<BdfCachedLinearSolver>>,
    /// Diagnostics from the most recent linear solve.
    last_linear_solve: RefCell<Option<BdfLinearSolveTelemetry>>,
    /// Right-hand sides for the two most recent solves. Consecutive affine
    /// probes use these to verify the accepted interpolated state directly.
    previous_linear_rhs: RefCell<Option<Vec<f64>>>,
    last_linear_rhs: RefCell<Option<Vec<f64>>>,
    last_linear_system: RefCell<Option<(sonium_solvers::CsrMatrix<f64>, Vec<f64>)>>,
}

impl BdfWorkspace {
    /// Assemble the constant blocks for the given mesh and spaces.
    pub fn new<VSpace: crate::spaces::FunctionSpace, PSpace: crate::spaces::FunctionSpace>(
        mesh: &crate::mesh::Mesh,
        v_space: &VSpace,
        p_space: &PSpace,
    ) -> Self {
        let n_v = v_space.num_global_dofs(mesh);
        let n_p = p_space.num_global_dofs(mesh);

        // Degree-aware quadrature: P1 keeps the legacy order 2 (bit-identical
        // numerics). P2 Oseen convection is degree 5: the advecting field and
        // test function are quadratic while the trial gradient is linear.
        let quad_order = if v_space.polynomial_degree() >= 2 {
            5
        } else {
            2
        };

        let velocity_dof_coordinates = v_space
            .dof_coordinates(mesh)
            .expect("BDF Navier-Stokes requires a nodal velocity space");
        assert_eq!(
            velocity_dof_coordinates.len(),
            n_v,
            "velocity space returned the wrong number of DOF coordinates"
        );
        let element_velocity_topology = ElementVelocityTopology::new(mesh, v_space);
        let mesh_centroid = mesh.nodes.iter().fold([0.0_f64; 3], |mut sum, point| {
            sum[0] += point.x;
            sum[1] += point.y;
            sum[2] += point.z;
            sum
        });
        let mesh_centroid = mesh_centroid.map(|sum| sum / mesh.num_nodes().max(1) as f64);
        let pressure_pin = p_space
            .dof_coordinates(mesh)
            .and_then(|coordinates| {
                coordinates
                    .iter()
                    .enumerate()
                    .min_by(|(_, a), (_, b)| {
                        let distance_a: f64 = (0..mesh.dimension)
                            .map(|direction| (a[direction] - mesh_centroid[direction]).powi(2))
                            .sum();
                        let distance_b: f64 = (0..mesh.dimension)
                            .map(|direction| (b[direction] - mesh_centroid[direction]).powi(2))
                            .sum();
                        distance_a.total_cmp(&distance_b)
                    })
                    .map(|(dof, _)| dof)
            })
            .unwrap_or(0);

        // Mass matrix for the time derivative.
        let mass_form = crate::forms::VectorMassForm { coefficient: 1.0 };
        let mass_triplet =
            crate::assembly::assemble_bilinear(&mass_form, v_space, v_space, mesh, quad_order);

        // Unit-viscosity Stokes stiffness; per-step scaling is `mu_step * K_unit`.
        let viscous_unit = super::forms::StokesViscousForm::new(1.0);
        let viscous_unit_triplet =
            crate::assembly::assemble_bilinear(&viscous_unit, v_space, v_space, mesh, quad_order);

        // Divergence coupling blocks.
        let system = crate::assembly::mixed::assemble_mixed(
            &super::forms::StokesViscousForm::new(0.0),
            &super::forms::StokesDivergenceForm,
            v_space,
            p_space,
            mesh,
            quad_order,
        );

        Self {
            mass_triplet,
            viscous_unit_triplet,
            block_01: system.block_01,
            block_10: system.block_10,
            n_v,
            n_p,
            pressure_pin,
            pressure_pin_value: 0.0,
            velocity_dof_coordinates,
            element_velocity_topology,
            quad_order,
            linear_solver_cache: RefCell::new(None),
            last_linear_solve: RefCell::new(None),
            previous_linear_rhs: RefCell::new(None),
            last_linear_rhs: RefCell::new(None),
            last_linear_system: RefCell::new(None),
        }
    }

    /// Set the pressure gauge used by subsequent cached BDF solves.
    ///
    /// Changing the gauge invalidates every matrix/RHS-dependent cache. For a
    /// traction boundary, choosing a pressure DOF on that boundary and using
    /// its prescribed pressure avoids the local distortion caused by an
    /// otherwise arbitrary interior point constraint.
    pub fn set_pressure_gauge(&mut self, pressure_dof: usize, pressure_value: f64) {
        assert!(
            pressure_dof < self.n_p,
            "pressure gauge DOF {pressure_dof} is out of range for {} pressure DOFs",
            self.n_p
        );
        assert!(
            pressure_value.is_finite(),
            "pressure gauge value must be finite"
        );
        self.pressure_pin = pressure_dof;
        self.pressure_pin_value = pressure_value;
        *self.linear_solver_cache.get_mut() = None;
        *self.last_linear_solve.get_mut() = None;
        *self.previous_linear_rhs.get_mut() = None;
        *self.last_linear_rhs.get_mut() = None;
        *self.last_linear_system.get_mut() = None;
    }

    /// Return diagnostics from the most recent cached BDF linear solve.
    pub fn last_linear_solve_telemetry(&self) -> Option<BdfLinearSolveTelemetry> {
        *self.last_linear_solve.borrow()
    }

    pub(crate) fn record_linear_solve(&self, telemetry: BdfLinearSolveTelemetry, rhs: &[f64]) {
        self.last_linear_solve.replace(Some(telemetry));
        let previous = self.last_linear_rhs.replace(Some(rhs.to_vec()));
        self.previous_linear_rhs.replace(previous);
    }

    /// Publish the exact boundary-conditioned BDF matrix and RHS used by the
    /// most recent cached solve.
    pub(crate) fn record_linear_system(
        &self,
        matrix: &sonium_solvers::CsrMatrix<f64>,
        rhs: &[f64],
    ) {
        self.last_linear_system
            .replace(Some((matrix.clone(), rhs.to_vec())));
    }

    /// Return the most recent exact cached BDF matrix/RHS pair.
    pub fn last_linear_system(&self) -> Option<(sonium_solvers::CsrMatrix<f64>, Vec<f64>)> {
        self.last_linear_system.borrow().clone()
    }

    /// Independently recompute the residual of an affine combination of the
    /// two most recently solved right-hand sides against the cached matrix.
    /// `factor=0` selects the previous RHS and `factor=1` the latest RHS.
    pub fn affine_true_relative_residual(&self, x: &[f64], factor: f64) -> Option<f64> {
        let cache = self.linear_solver_cache.borrow();
        let matrix = match cache.as_ref()? {
            BdfCachedLinearSolver::Direct { matrix, .. }
            | BdfCachedLinearSolver::JacobiSchur { matrix, .. }
            | BdfCachedLinearSolver::Ilu { matrix, .. } => matrix,
        };
        let previous = self.previous_linear_rhs.borrow();
        let latest = self.last_linear_rhs.borrow();
        let previous = previous.as_ref()?;
        let latest = latest.as_ref()?;
        if previous.len() != latest.len()
            || previous.len() != matrix.num_rows
            || x.len() != matrix.num_cols
            || !factor.is_finite()
        {
            return None;
        }
        let ax = matrix.matvec(x);
        let (residual_sq, rhs_sq) = ax.iter().zip(previous).zip(latest).fold(
            (0.0, 0.0),
            |(residual_sq, rhs_sq), ((a, base), probe)| {
                let rhs = base + factor * (probe - base);
                (residual_sq + (rhs - a).powi(2), rhs_sq + rhs * rhs)
            },
        );
        Some(residual_sq.sqrt() / rhs_sq.sqrt().max(f64::MIN_POSITIVE))
    }
}

const BARYCENTRIC_TOLERANCE: f64 = 1.0e-9;

#[derive(Debug)]
struct VelocityElement {
    element_type: ElementType,
    vertices: [[f64; 3]; 4],
    vertex_count: usize,
    scalar_dofs: [usize; 10],
    support_count: usize,
    bbox_min: [f64; 3],
    bbox_max: [f64; 3],
}

const BVH_LEAF_SIZE: usize = 8;

#[derive(Debug)]
enum VelocityBvhKind {
    Branch,
    Leaf(Vec<usize>),
}

/// Pre-order, stackless AABB-BVH node. `escape` is the first node after this
/// subtree, so a rejected AABB can skip the entire subtree without allocating
/// a traversal stack at every quadrature point.
#[derive(Debug)]
struct VelocityBvhNode {
    bbox_min: [f64; 3],
    bbox_max: [f64; 3],
    escape: usize,
    kind: VelocityBvhKind,
}

/// Cached bounded AABB BVH and local DOF maps for exact P1/P2 velocity lookup.
#[derive(Debug)]
pub(crate) struct ElementVelocityTopology {
    elements: Vec<VelocityElement>,
    bvh: Vec<VelocityBvhNode>,
    components: usize,
    degree: usize,
    required_velocity_len: usize,
}

/// Borrowing view of one extrapolated velocity vector over cached topology.
pub(crate) struct ElementVelocityField<'a> {
    topology: &'a ElementVelocityTopology,
    values: &'a [f64],
}

impl ElementVelocityTopology {
    fn new<VSpace: crate::spaces::FunctionSpace>(
        mesh: &crate::mesh::Mesh,
        v_space: &VSpace,
    ) -> Self {
        let degree = v_space.polynomial_degree();
        assert!(
            matches!(degree, 1 | 2),
            "BDF velocity evaluation supports only P1/P2 spaces"
        );
        let components = v_space.num_components();
        assert_eq!(
            components, mesh.dimension,
            "BDF velocity components must match mesh dimension"
        );

        let mut elements = Vec::with_capacity(mesh.elements.len());
        for (elem_idx, element) in mesh.elements.iter().enumerate() {
            assert!(
                matches!(
                    element.element_type,
                    ElementType::Triangle | ElementType::Tetrahedron
                ),
                "BDF velocity evaluation supports only triangle/tetrahedron elements"
            );
            let vertex_count = element.element_type.num_vertices();
            let support_count = match (element.element_type, degree) {
                (ElementType::Triangle, 1) => 3,
                (ElementType::Triangle, 2) => 6,
                (ElementType::Tetrahedron, 1) => 4,
                (ElementType::Tetrahedron, 2) => 10,
                _ => unreachable!(),
            };
            let mut vertices = [[0.0; 3]; 4];
            let mut bbox_min = [f64::INFINITY; 3];
            let mut bbox_max = [f64::NEG_INFINITY; 3];
            for (slot, &node) in element.vertices().iter().enumerate() {
                let point = mesh
                    .nodes
                    .get(node)
                    .expect("element references missing mesh node");
                let coords = [point.x, point.y, point.z];
                assert!(
                    coords.iter().all(|x| x.is_finite()),
                    "BDF velocity mesh contains non-finite coordinates"
                );
                vertices[slot] = coords;
                for axis in 0..3 {
                    bbox_min[axis] = bbox_min[axis].min(coords[axis]);
                    bbox_max[axis] = bbox_max[axis].max(coords[axis]);
                }
            }
            let local = v_space.local_to_global(mesh, elem_idx);
            assert_eq!(
                local.len(),
                support_count * components,
                "velocity element local DOFs do not match its P1/P2 layout"
            );
            let mut scalar_dofs = [0; 10];
            for support in 0..support_count {
                let base = local[support * components];
                assert_eq!(
                    base % components,
                    0,
                    "velocity DOFs must use interleaved component ordering"
                );
                let scalar = base / components;
                for component in 0..components {
                    assert_eq!(
                        local[support * components + component],
                        scalar * components + component,
                        "velocity DOFs must use interleaved component ordering"
                    );
                }
                scalar_dofs[support] = scalar;
            }
            assert!(
                simplex_is_nondegenerate(element.element_type, &vertices),
                "BDF velocity mesh contains a degenerate element"
            );
            elements.push(VelocityElement {
                element_type: element.element_type,
                vertices,
                vertex_count,
                scalar_dofs,
                support_count,
                bbox_min,
                bbox_max,
            });
        }

        let mut bvh = Vec::with_capacity(elements.len().saturating_mul(2));
        if !elements.is_empty() {
            let mut indices: Vec<usize> = (0..elements.len()).collect();
            build_velocity_bvh(&elements, &mut indices, &mut bvh);
        }
        Self {
            elements,
            bvh,
            components,
            degree,
            required_velocity_len: v_space.num_global_dofs(mesh),
        }
    }

    pub(crate) fn field<'a>(&'a self, values: &'a [f64]) -> ElementVelocityField<'a> {
        ElementVelocityField {
            topology: self,
            values,
        }
    }
}

fn build_velocity_bvh(
    elements: &[VelocityElement],
    indices: &mut [usize],
    nodes: &mut Vec<VelocityBvhNode>,
) {
    let mut bbox_min = [f64::INFINITY; 3];
    let mut bbox_max = [f64::NEG_INFINITY; 3];
    for &index in indices.iter() {
        for axis in 0..3 {
            bbox_min[axis] = bbox_min[axis].min(elements[index].bbox_min[axis]);
            bbox_max[axis] = bbox_max[axis].max(elements[index].bbox_max[axis]);
        }
    }

    let node_index = nodes.len();
    nodes.push(VelocityBvhNode {
        bbox_min,
        bbox_max,
        escape: 0,
        kind: VelocityBvhKind::Branch,
    });
    if indices.len() <= BVH_LEAF_SIZE {
        indices.sort_unstable();
        nodes[node_index].kind = VelocityBvhKind::Leaf(indices.to_vec());
    } else {
        let axis = (0..3)
            .max_by(|&a, &b| (bbox_max[a] - bbox_min[a]).total_cmp(&(bbox_max[b] - bbox_min[b])))
            .unwrap();
        indices.sort_by(|&a, &b| {
            let a_center = elements[a].bbox_min[axis] + elements[a].bbox_max[axis];
            let b_center = elements[b].bbox_min[axis] + elements[b].bbox_max[axis];
            a_center.total_cmp(&b_center).then(a.cmp(&b))
        });
        let (left, right) = indices.split_at_mut(indices.len() / 2);
        build_velocity_bvh(elements, left, nodes);
        build_velocity_bvh(elements, right, nodes);
    }
    nodes[node_index].escape = nodes.len();
}

fn bbox_contains(node: &VelocityBvhNode, point: [f64; 3]) -> bool {
    let scale = (0..3)
        .map(|axis| node.bbox_max[axis] - node.bbox_min[axis])
        .fold(0.0_f64, f64::max);
    let tolerance = BARYCENTRIC_TOLERANCE * scale;
    (0..3).all(|axis| {
        point[axis] >= node.bbox_min[axis] - tolerance
            && point[axis] <= node.bbox_max[axis] + tolerance
    })
}

impl ElementVelocityField<'_> {
    pub(crate) fn velocity_at(&self, point: [f64; 3]) -> Option<[f64; 3]> {
        self.velocity_at_with_candidate_count(point).0
    }

    fn velocity_at_with_candidate_count(&self, point: [f64; 3]) -> (Option<[f64; 3]>, usize) {
        let topology = self.topology;
        if self.values.len() != topology.required_velocity_len
            || !point.iter().all(|x| x.is_finite())
        {
            return (None, 0);
        }
        let mut node_index = 0;
        let mut candidate_count = 0;
        let mut located = None;
        while node_index < topology.bvh.len() {
            let node = &topology.bvh[node_index];
            if !bbox_contains(node, point) {
                node_index = node.escape;
                continue;
            }
            match &node.kind {
                VelocityBvhKind::Branch => node_index += 1,
                VelocityBvhKind::Leaf(indices) => {
                    for &index in indices {
                        let element = &topology.elements[index];
                        candidate_count += 1;
                        if (0..3).any(|axis| {
                            let scale = element.bbox_max[axis] - element.bbox_min[axis];
                            let tolerance = BARYCENTRIC_TOLERANCE * scale;
                            point[axis] < element.bbox_min[axis] - tolerance
                                || point[axis] > element.bbox_max[axis] + tolerance
                        }) {
                            continue;
                        }
                        let barycentric = match element.element_type {
                            ElementType::Triangle => triangle_barycentric(
                                &element.vertices[..element.vertex_count],
                                point,
                            ),
                            ElementType::Tetrahedron => tetrahedron_barycentric(
                                &element.vertices[..element.vertex_count],
                                point,
                            ),
                            _ => unreachable!(),
                        };
                        if barycentric.is_some() && located.is_none_or(|best| index < best) {
                            located = Some(index);
                        }
                    }
                    node_index += 1;
                }
            }
        }
        let Some(index) = located else {
            return (None, candidate_count);
        };
        let element = &topology.elements[index];
        let barycentric = match element.element_type {
            ElementType::Triangle => {
                triangle_barycentric(&element.vertices[..element.vertex_count], point)
            }
            ElementType::Tetrahedron => {
                tetrahedron_barycentric(&element.vertices[..element.vertex_count], point)
            }
            _ => unreachable!(),
        }
        .expect("located BVH element must contain the query point");
        let mut weights = [0.0; 10];
        if topology.degree == 1 {
            weights[..element.support_count].copy_from_slice(&barycentric[..element.support_count]);
        } else {
            match element.element_type {
                ElementType::Triangle => weights[..6]
                    .copy_from_slice(&crate::basis::p2_triangle(barycentric[1], barycentric[2])),
                ElementType::Tetrahedron => weights.copy_from_slice(&crate::basis::p2_tetrahedron(
                    barycentric[1],
                    barycentric[2],
                    barycentric[3],
                )),
                _ => unreachable!(),
            }
        }
        let mut velocity = [0.0; 3];
        for (support, &w) in weights[..element.support_count].iter().enumerate() {
            for (component, value) in velocity.iter_mut().enumerate().take(topology.components) {
                *value +=
                    w * self.values[element.scalar_dofs[support] * topology.components + component];
            }
        }
        (Some(velocity), candidate_count)
    }
}

fn simplex_is_nondegenerate(element_type: ElementType, vertices: &[[f64; 3]; 4]) -> bool {
    let edge_scale = |count: usize| {
        let mut scale = 0.0_f64;
        for i in 0..count {
            for j in i + 1..count {
                let distance = (0..3)
                    .map(|axis| (vertices[i][axis] - vertices[j][axis]).powi(2))
                    .sum::<f64>()
                    .sqrt();
                scale = scale.max(distance);
            }
        }
        scale
    };
    match element_type {
        ElementType::Triangle => {
            let [a, b, c, _] = *vertices;
            let determinant = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]);
            let scale = edge_scale(3);
            determinant.is_finite()
                && scale.is_finite()
                && scale > 0.0
                && determinant.abs() > 64.0 * f64::EPSILON * scale.powi(2)
        }
        ElementType::Tetrahedron => {
            let [a, b, c, d] = *vertices;
            let det = |u: [f64; 3], v: [f64; 3], w: [f64; 3]| {
                u[0] * (v[1] * w[2] - v[2] * w[1]) - u[1] * (v[0] * w[2] - v[2] * w[0])
                    + u[2] * (v[0] * w[1] - v[1] * w[0])
            };
            let ab = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
            let ac = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
            let ad = [d[0] - a[0], d[1] - a[1], d[2] - a[2]];
            let determinant = det(ab, ac, ad);
            let scale = edge_scale(4);
            determinant.is_finite()
                && scale.is_finite()
                && scale > 0.0
                && determinant.abs() > 64.0 * f64::EPSILON * scale.powi(3)
        }
        _ => false,
    }
}

fn triangle_barycentric(vertices: &[[f64; 3]], point: [f64; 3]) -> Option<[f64; 4]> {
    let [a, b, c]: [[f64; 3]; 3] = vertices.try_into().ok()?;
    let v0 = [b[0] - a[0], b[1] - a[1]];
    let v1 = [c[0] - a[0], c[1] - a[1]];
    let v2 = [point[0] - a[0], point[1] - a[1]];
    let denom = v0[0] * v1[1] - v0[1] * v1[0];
    if !denom.is_finite() || denom.abs() <= f64::EPSILON {
        return None;
    }
    let l1 = (v2[0] * v1[1] - v2[1] * v1[0]) / denom;
    let l2 = (v0[0] * v2[1] - v0[1] * v2[0]) / denom;
    let result = [1.0 - l1 - l2, l1, l2, 0.0];
    result[..3]
        .iter()
        .all(|&x| x >= -BARYCENTRIC_TOLERANCE)
        .then_some(result)
}

fn tetrahedron_barycentric(vertices: &[[f64; 3]], point: [f64; 3]) -> Option<[f64; 4]> {
    let [a, b, c, d]: [[f64; 3]; 4] = vertices.try_into().ok()?;
    let det = |u: [f64; 3], v: [f64; 3], w: [f64; 3]| {
        u[0] * (v[1] * w[2] - v[2] * w[1]) - u[1] * (v[0] * w[2] - v[2] * w[0])
            + u[2] * (v[0] * w[1] - v[1] * w[0])
    };
    let ab = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
    let ac = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
    let ad = [d[0] - a[0], d[1] - a[1], d[2] - a[2]];
    let ap = [point[0] - a[0], point[1] - a[1], point[2] - a[2]];
    let denominator = det(ab, ac, ad);
    if !denominator.is_finite() || denominator.abs() <= f64::EPSILON {
        return None;
    }
    let result = [
        1.0 - (det(ap, ac, ad) + det(ab, ap, ad) + det(ab, ac, ap)) / denominator,
        det(ap, ac, ad) / denominator,
        det(ab, ap, ad) / denominator,
        det(ab, ac, ap) / denominator,
    ];
    result
        .iter()
        .all(|&x| x >= -BARYCENTRIC_TOLERANCE)
        .then_some(result)
}

/// Compatibility helper for callers that explicitly need nearest-node lookup.
/// BDF convection uses [`ElementVelocityTopology`] and exact FE interpolation.
pub struct NodeGrid {
    /// Cell index → node indices contained in the cell.
    cells: std::collections::HashMap<(i32, i32, i32), Vec<usize>>,
    /// Grid cell edge length.
    cell_size: f64,
    /// Copy of the node coordinates (queries return indices into this).
    coords: Vec<[f64; 3]>,
}

impl NodeGrid {
    /// Build a grid over `coords` with the given cell edge length. A
    /// non-positive or non-finite `cell_size` degenerates to a full scan.
    pub fn build(coords: &[[f64; 3]], cell_size: f64) -> Self {
        let mut cells: std::collections::HashMap<(i32, i32, i32), Vec<usize>> =
            std::collections::HashMap::new();
        if cell_size.is_finite() && cell_size > 0.0 {
            for (i, p) in coords.iter().enumerate() {
                cells
                    .entry(Self::cell_of(p, cell_size))
                    .or_default()
                    .push(i);
            }
        }
        Self {
            cells,
            cell_size,
            coords: coords.to_vec(),
        }
    }

    /// Index of the node closest to `p` (ties broken toward the lower index).
    pub fn nearest(&self, p: [f64; 3]) -> usize {
        if self.cells.is_empty() {
            return self.full_scan(p);
        }
        let q = Self::cell_of(&p, self.cell_size);
        let cs = self.cell_size;
        let mut best_idx = 0usize;
        let mut best_dist = f64::MAX;
        let mut ring = 0i32;
        loop {
            let mut searched_any = false;
            for dx in -ring..=ring {
                for dy in -ring..=ring {
                    for dz in -ring..=ring {
                        // Only the shell at Chebyshev distance `ring`.
                        if ring > 0 && dx.abs().max(dy.abs()).max(dz.abs()) != ring {
                            continue;
                        }
                        if let Some(bucket) = self.cells.get(&(q.0 + dx, q.1 + dy, q.2 + dz)) {
                            searched_any = true;
                            for &i in bucket {
                                let c = &self.coords[i];
                                let d = (p[0] - c[0]).powi(2)
                                    + (p[1] - c[1]).powi(2)
                                    + (p[2] - c[2]).powi(2);
                                if d < best_dist || (d == best_dist && i < best_idx) {
                                    best_dist = d;
                                    best_idx = i;
                                }
                            }
                        }
                    }
                }
            }
            // Searched box [q-ring, q+ring+1)·cs. Any node in an unsearched
            // cell lies at distance >= escape from `p`, where `escape` is the
            // distance from `p` to the outside of that box. Stop once the best
            // candidate is strictly closer than any possible unsearched node.
            let lo = [
                (q.0 - ring) as f64 * cs,
                (q.1 - ring) as f64 * cs,
                (q.2 - ring) as f64 * cs,
            ];
            let hi = [
                (q.0 + ring + 1) as f64 * cs,
                (q.1 + ring + 1) as f64 * cs,
                (q.2 + ring + 1) as f64 * cs,
            ];
            let mut escape = f64::MAX;
            for a in 0..3 {
                let da = if p[a] < lo[a] {
                    lo[a] - p[a]
                } else if p[a] > hi[a] {
                    p[a] - hi[a]
                } else {
                    (p[a] - lo[a]).min(hi[a] - p[a])
                };
                escape = escape.min(da);
            }
            if best_dist < escape * escape {
                return best_idx;
            }
            // Degenerate neighborhood (no nodes found yet and no closer box
            // boundary to cut on): fall back to a full scan rather than
            // expanding rings forever.
            if !searched_any && ring > 0 && best_dist == f64::MAX && self.ring_exhausted(q, ring) {
                return self.full_scan(p);
            }
            ring += 1;
        }
    }

    /// True once the shell at `ring` can no longer intersect any stored cell:
    /// every cell key lies inside the already-searched box.
    fn ring_exhausted(&self, q: (i32, i32, i32), ring: i32) -> bool {
        self.cells.keys().all(|&(cx, cy, cz)| {
            (cx - q.0).abs().max((cy - q.1).abs()).max((cz - q.2).abs()) <= ring
        })
    }

    fn full_scan(&self, p: [f64; 3]) -> usize {
        let mut best_idx = 0usize;
        let mut best_dist = f64::MAX;
        for (i, c) in self.coords.iter().enumerate() {
            let d = (p[0] - c[0]).powi(2) + (p[1] - c[1]).powi(2) + (p[2] - c[2]).powi(2);
            if d < best_dist {
                best_dist = d;
                best_idx = i;
            }
        }
        best_idx
    }

    fn cell_of(p: &[f64; 3], cell_size: f64) -> (i32, i32, i32) {
        (
            (p[0] / cell_size).floor() as i32,
            (p[1] / cell_size).floor() as i32,
            (p[2] / cell_size).floor() as i32,
        )
    }
}

/// Maximum element edge length over the mesh (longest node-to-node distance
/// within any element). Returns 0.0 for an empty mesh.
pub fn max_edge_length(mesh: &crate::mesh::Mesh) -> f64 {
    let mut max_d2 = 0.0f64;
    for elem in &mesh.elements {
        for (i, &a) in elem.nodes.iter().enumerate() {
            let pa = &mesh.nodes[a];
            for &b in &elem.nodes[i + 1..] {
                let pb = &mesh.nodes[b];
                let d2 = (pa.x - pb.x).powi(2) + (pa.y - pb.y).powi(2) + (pa.z - pb.z).powi(2);
                if d2 > max_d2 {
                    max_d2 = d2;
                }
            }
        }
    }
    max_d2.sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::basis::PolynomialDegree;
    use crate::mesh::{unit_cube_tetrahedra, unit_square_triangles};
    use crate::spaces::{FunctionSpace, H1Space, VectorH1Space};

    #[test]
    fn p2_bdf_workspace_uses_degree_five_oseen_quadrature() {
        let mesh = unit_cube_tetrahedra(1);
        let velocity_space = VectorH1Space::new(PolynomialDegree::P2, 3);
        let pressure_space = H1Space::new(PolynomialDegree::P1);
        let workspace = BdfWorkspace::new(&mesh, &velocity_space, &pressure_space);

        // w(P2) · grad(u)(P1) · v(P2) is polynomial degree five. Degree four
        // is sufficient for mass but under-integrates the nonlinear operator.
        assert_eq!(workspace.quad_order, 5);
    }

    #[test]
    fn bdf_pressure_gauge_accepts_prescribed_boundary_value() {
        let mesh = unit_cube_tetrahedra(1);
        let velocity_space = VectorH1Space::new(PolynomialDegree::P2, 3);
        let pressure_space = H1Space::new(PolynomialDegree::P1);
        let mut workspace = BdfWorkspace::new(&mesh, &velocity_space, &pressure_space);

        workspace.set_pressure_gauge(1, 17.0);

        assert_eq!(workspace.pressure_pin, 1);
        assert_eq!(workspace.pressure_pin_value, 17.0);
    }

    fn assert_velocity_mass_has_scalar_component_blocks(degree: PolynomialDegree) {
        let mesh = unit_square_triangles(2);
        let scalar_space = H1Space::new(degree);
        let velocity_space = VectorH1Space::new(degree, 2);
        let pressure_space = H1Space::new(PolynomialDegree::P1);
        let workspace = BdfWorkspace::new(&mesh, &velocity_space, &pressure_space);

        let scalar_mass = crate::assembly::assemble_bilinear(
            &crate::forms::MassForm,
            &scalar_space,
            &scalar_space,
            &mesh,
            workspace.quad_order,
        )
        .to_csr();
        let velocity_mass = workspace.mass_triplet.to_csr();
        let n_scalar = scalar_space.num_global_dofs(&mesh);

        assert_eq!(velocity_mass.num_rows, 2 * n_scalar);
        assert_eq!(velocity_mass.num_cols, 2 * n_scalar);

        for scalar_row in 0..n_scalar {
            for scalar_col in 0..n_scalar {
                let scalar_value = scalar_mass
                    .row_entries(scalar_row)
                    .find(|(col, _)| *col == scalar_col)
                    .map_or(0.0, |(_, value)| value);

                for row_component in 0..2 {
                    for col_component in 0..2 {
                        let row = scalar_row * 2 + row_component;
                        let col = scalar_col * 2 + col_component;
                        let velocity_value = velocity_mass
                            .row_entries(row)
                            .find(|(entry_col, _)| *entry_col == col)
                            .map_or(0.0, |(_, value)| value);
                        if row_component == col_component {
                            assert!(
                                (velocity_value - scalar_value).abs() < 1.0e-13,
                                "{degree:?} velocity mass ({row}, {col}) = {velocity_value:e}, expected {scalar_value:e}"
                            );
                        } else {
                            assert_eq!(
                                velocity_value, 0.0,
                                "{degree:?} cross-component velocity mass ({row}, {col}) must be structurally zero"
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn p1_velocity_mass_has_independent_scalar_component_blocks() {
        assert_velocity_mass_has_scalar_component_blocks(PolynomialDegree::P1);
    }

    #[test]
    fn p2_velocity_mass_has_independent_scalar_component_blocks() {
        assert_velocity_mass_has_scalar_component_blocks(PolynomialDegree::P2);
    }

    fn fill_field(
        mesh: &crate::mesh::Mesh,
        space: &VectorH1Space,
        evaluate: impl Fn([f64; 3]) -> [f64; 3],
    ) -> Vec<f64> {
        (0..space.num_global_dofs(mesh))
            .map(|dof| {
                let point = space.dof_coordinate(mesh, dof).unwrap();
                evaluate(point)[dof % space.num_components()]
            })
            .collect()
    }

    fn element_point(element: &VelocityElement, barycentric: &[f64]) -> [f64; 3] {
        let mut point = [0.0; 3];
        for (vertex, weight) in element.vertices[..element.vertex_count]
            .iter()
            .zip(barycentric)
        {
            for axis in 0..3 {
                point[axis] += vertex[axis] * weight;
            }
        }
        point
    }

    fn brute_force_element(topology: &ElementVelocityTopology, point: [f64; 3]) -> Option<usize> {
        topology
            .elements
            .iter()
            .position(|element| match element.element_type {
                ElementType::Triangle => {
                    triangle_barycentric(&element.vertices[..3], point).is_some()
                }
                ElementType::Tetrahedron => {
                    tetrahedron_barycentric(&element.vertices[..4], point).is_some()
                }
                _ => false,
            })
    }

    fn accelerated_element(topology: &ElementVelocityTopology, point: [f64; 3]) -> Option<usize> {
        let mut node_index = 0;
        let mut located = None;
        while node_index < topology.bvh.len() {
            let node = &topology.bvh[node_index];
            if !bbox_contains(node, point) {
                node_index = node.escape;
                continue;
            }
            match &node.kind {
                VelocityBvhKind::Branch => node_index += 1,
                VelocityBvhKind::Leaf(indices) => {
                    for &index in indices {
                        let element = &topology.elements[index];
                        let contains = match element.element_type {
                            ElementType::Triangle => {
                                triangle_barycentric(&element.vertices[..3], point).is_some()
                            }
                            ElementType::Tetrahedron => {
                                tetrahedron_barycentric(&element.vertices[..4], point).is_some()
                            }
                            _ => false,
                        };
                        if contains && located.is_none_or(|best| index < best) {
                            located = Some(index);
                        }
                    }
                    node_index += 1;
                }
            }
        }
        located
    }

    fn assert_locator_matches_brute_force(mesh: crate::mesh::Mesh, degree: PolynomialDegree) {
        let space = VectorH1Space::new(degree, mesh.dimension);
        let topology = ElementVelocityTopology::new(&mesh, &space);
        for element in &topology.elements {
            let barycentric: &[f64] = if mesh.dimension == 2 {
                &[0.2, 0.3, 0.5]
            } else {
                &[0.1, 0.2, 0.3, 0.4]
            };
            let point = element_point(element, barycentric);
            assert_eq!(
                accelerated_element(&topology, point),
                brute_force_element(&topology, point)
            );
        }
        // Shared-boundary deterministic lowest-index selection.
        let point = topology.elements[0].vertices[0];
        assert_eq!(
            accelerated_element(&topology, point),
            brute_force_element(&topology, point)
        );
    }

    #[test]
    fn element_velocity_locator_matches_brute_force_for_triangles() {
        assert_locator_matches_brute_force(unit_square_triangles(5), PolynomialDegree::P2);
    }

    #[test]
    fn element_velocity_locator_matches_brute_force_for_tetrahedra() {
        assert_locator_matches_brute_force(unit_cube_tetrahedra(3), PolynomialDegree::P2);
    }

    #[test]
    fn element_velocity_locator_returns_none_outside_domain() {
        let mesh = unit_square_triangles(2);
        let space = VectorH1Space::new(PolynomialDegree::P2, 2);
        let topology = ElementVelocityTopology::new(&mesh, &space);
        let values = vec![0.0; space.num_global_dofs(&mesh)];
        assert_eq!(topology.field(&values).velocity_at([-1.0, 0.5, 0.0]), None);
        assert_eq!(
            topology
                .field(&values[..values.len() - 1])
                .velocity_at([0.2, 0.2, 0.0]),
            None
        );
        assert_eq!(
            topology.field(&values).velocity_at([f64::NAN, 0.2, 0.0]),
            None
        );
    }

    #[test]
    fn element_velocity_topology_rejects_degenerate_simplices_at_construction() {
        use crate::mesh::{Mesh, Point};

        let mut triangle = Mesh::new(2);
        triangle.nodes = vec![
            Point::new_2d(0.0, 0.0),
            Point::new_2d(1.0, 0.0),
            Point::new_2d(2.0, 0.0),
        ];
        triangle.add_element(ElementType::Triangle, vec![0, 1, 2]);
        let triangle_space = VectorH1Space::new(PolynomialDegree::P1, 2);
        assert!(
            std::panic::catch_unwind(|| {
                ElementVelocityTopology::new(&triangle, &triangle_space)
            })
            .is_err()
        );

        let mut tetrahedron = Mesh::new(3);
        tetrahedron.nodes = vec![
            Point::new_3d(0.0, 0.0, 0.0),
            Point::new_3d(1.0, 0.0, 0.0),
            Point::new_3d(0.0, 1.0, 0.0),
            Point::new_3d(1.0, 1.0, 0.0),
        ];
        tetrahedron.add_element(ElementType::Tetrahedron, vec![0, 1, 2, 3]);
        let tetrahedron_space = VectorH1Space::new(PolynomialDegree::P1, 3);
        assert!(
            std::panic::catch_unwind(|| {
                ElementVelocityTopology::new(&tetrahedron, &tetrahedron_space)
            })
            .is_err()
        );
    }

    fn assert_exact_field(mesh: crate::mesh::Mesh, degree: PolynomialDegree) {
        let dim = mesh.dimension;
        let space = VectorH1Space::new(degree, dim);
        let topology = ElementVelocityTopology::new(&mesh, &space);
        let affine = |p: [f64; 3]| {
            [
                1.0 + 2.0 * p[0] - 3.0 * p[1] + p[2],
                -2.0 + 0.5 * p[0] + 4.0 * p[1] - 2.0 * p[2],
                4.0 - p[0] + 3.0 * p[1] + 0.25 * p[2],
            ]
        };
        let values = fill_field(&mesh, &space, affine);
        let field = topology.field(&values);
        for element in &topology.elements {
            let barycentric: &[f64] = if dim == 2 {
                &[0.17, 0.31, 0.52]
            } else {
                &[0.13, 0.19, 0.29, 0.39]
            };
            let point = element_point(element, barycentric);
            let actual = field.velocity_at(point).unwrap();
            let expected = affine(point);
            for component in 0..dim {
                assert!((actual[component] - expected[component]).abs() < 1.0e-12);
            }
        }
        if degree == PolynomialDegree::P2 {
            let quadratic = |p: [f64; 3]| {
                [
                    p[0] * p[0],
                    p[0] * p[1] + p[1] * p[1],
                    p[1] * p[2] + p[2] * p[2],
                ]
            };
            let values = fill_field(&mesh, &space, quadratic);
            let field = topology.field(&values);
            for element in &topology.elements {
                let barycentric: &[f64] = if dim == 2 {
                    &[0.23, 0.34, 0.43]
                } else {
                    &[0.11, 0.22, 0.28, 0.39]
                };
                let point = element_point(element, barycentric);
                let actual = field.velocity_at(point).unwrap();
                let expected = quadratic(point);
                for component in 0..dim {
                    assert!((actual[component] - expected[component]).abs() < 1.0e-12);
                }
            }
        }
    }

    #[test]
    fn p1_triangle_element_velocity_reproduces_affine_field() {
        assert_exact_field(unit_square_triangles(3), PolynomialDegree::P1);
    }
    #[test]
    fn p2_triangle_element_velocity_reproduces_affine_and_quadratic_fields() {
        assert_exact_field(unit_square_triangles(3), PolynomialDegree::P2);
    }
    #[test]
    fn p1_tetrahedron_element_velocity_reproduces_affine_field() {
        assert_exact_field(unit_cube_tetrahedra(2), PolynomialDegree::P1);
    }
    #[test]
    fn p2_tetrahedron_element_velocity_reproduces_affine_and_quadratic_fields() {
        assert_exact_field(unit_cube_tetrahedra(2), PolynomialDegree::P2);
    }

    fn lookup_stats(mesh: crate::mesh::Mesh) -> (f64, usize) {
        let space = VectorH1Space::new(PolynomialDegree::P2, mesh.dimension);
        let topology = ElementVelocityTopology::new(&mesh, &space);
        let values = vec![0.0; space.num_global_dofs(&mesh)];
        let field = topology.field(&values);
        let mut total = 0;
        let mut max = 0;
        for element in &topology.elements {
            let barycentric: &[f64] = if mesh.dimension == 2 {
                &[0.2, 0.3, 0.5]
            } else {
                &[0.1, 0.2, 0.3, 0.4]
            };
            let count = field
                .velocity_at_with_candidate_count(element_point(element, barycentric))
                .1;
            total += count;
            max = max.max(count);
        }
        (total as f64 / topology.elements.len() as f64, max)
    }

    #[test]
    fn element_velocity_lookup_candidate_count_stays_bounded_under_refinement() {
        let (tri_coarse, tri_coarse_max) = lookup_stats(unit_square_triangles(8));
        let (tri_fine, tri_fine_max) = lookup_stats(unit_square_triangles(16));
        assert!(
            tri_coarse < 16.0 && tri_fine < 16.0 && tri_coarse_max < 64 && tri_fine_max < 64,
            "triangle candidates coarse={tri_coarse}/{tri_coarse_max}, fine={tri_fine}/{tri_fine_max}"
        );
        let (tet_coarse, tet_coarse_max) = lookup_stats(unit_cube_tetrahedra(3));
        let (tet_fine, tet_fine_max) = lookup_stats(unit_cube_tetrahedra(6));
        assert!(
            tet_coarse < 48.0 && tet_fine < 48.0 && tet_coarse_max <= 48 && tet_fine_max <= 48,
            "tet candidates coarse={tet_coarse}/{tet_coarse_max}, fine={tet_fine}/{tet_fine_max}"
        );
        assert!(tri_fine < tri_coarse * 2.0 && tet_fine < tet_coarse * 2.0);
    }

    #[test]
    fn element_velocity_bvh_handles_strongly_graded_mesh_with_bounded_candidates() {
        use crate::mesh::{Mesh, Point};

        let mut mesh = Mesh::new(2);
        mesh.nodes = vec![
            Point::new_2d(0.0, 0.0),
            Point::new_2d(1.0, 0.0),
            Point::new_2d(0.0, 1.0),
        ];
        mesh.add_element(ElementType::Triangle, vec![0, 1, 2]);
        let tiny = 1.0e-6;
        for i in 0..64 {
            let x = 2.0 + 2.0 * tiny * i as f64;
            let first = mesh.nodes.len();
            mesh.nodes.extend([
                Point::new_2d(x, 0.0),
                Point::new_2d(x + tiny, 0.0),
                Point::new_2d(x, tiny),
            ]);
            mesh.add_element(ElementType::Triangle, vec![first, first + 1, first + 2]);
        }

        let space = VectorH1Space::new(PolynomialDegree::P1, 2);
        let topology = ElementVelocityTopology::new(&mesh, &space);
        let values = vec![0.0; space.num_global_dofs(&mesh)];
        let field = topology.field(&values);
        let (velocity, coarse_candidates) =
            field.velocity_at_with_candidate_count([0.25, 0.25, 0.0]);
        assert_eq!(velocity, Some([0.0; 3]));
        assert!(
            coarse_candidates <= 2 * BVH_LEAF_SIZE,
            "graded-mesh coarse query tested {coarse_candidates} candidates"
        );
        let (velocity, tiny_candidates) =
            field.velocity_at_with_candidate_count([2.0 + tiny / 4.0, tiny / 4.0, 0.0]);
        assert_eq!(velocity, Some([0.0; 3]));
        assert!(
            tiny_candidates <= 2 * BVH_LEAF_SIZE,
            "graded-mesh tiny query tested {tiny_candidates} candidates"
        );
        assert!(topology.bvh.len() < 2 * topology.elements.len());
    }

    #[test]
    fn node_grid_nearest_matches_brute_force() {
        let mut rng_state = 0x12345678u64;
        let mut rand = || {
            rng_state ^= rng_state << 13;
            rng_state ^= rng_state >> 7;
            rng_state ^= rng_state << 17;
            (rng_state % 10000) as f64 / 10000.0
        };
        let coords: Vec<[f64; 3]> = (0..500).map(|_| [rand(), rand(), rand()]).collect();
        let grid = NodeGrid::build(&coords, 0.1);
        for _ in 0..200 {
            let p = [rand(), rand(), rand()];
            let brute = coords
                .iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| {
                    let da = (a[0] - p[0]).powi(2) + (a[1] - p[1]).powi(2) + (a[2] - p[2]).powi(2);
                    let db = (b[0] - p[0]).powi(2) + (b[1] - p[1]).powi(2) + (b[2] - p[2]).powi(2);
                    da.partial_cmp(&db).unwrap()
                })
                .unwrap()
                .0;
            assert_eq!(grid.nearest(p), brute);
        }
    }
}
