//! Explicit TETRA4 and PENTA6 solid elements with C1 materials.
//!
//! Constant-strain [`Tet4Geom`] tetrahedra (Theory Manual "Solid
//! Tetrahedron Elements", p. 101) integrate exactly with one material
//! point, while [`Penta6Geom`] wedges use full 6-point quadrature (3
//! triangle points by 2 through-thickness Gauss points), the degenerate
//! brick of "Element Degeneration" (p. 92). Both evaluate explicit
//! internal forces `f = sum_q w_q B_q^T (sigma_q + q I)` with the
//! stress from any frozen C1 [`StressUpdateModel`] and the shock
//! viscosity `q` of "Shock Waves" (pp. 91-92, Eq 340). Full quadrature
//! leaves no hourglass modes, so no stabilization enters; reduced
//! wedge quadrature with stabilization is deferred. Lumped masses and
//! the `L / c` critical step follow "Time Step" (p. 103).
//!
//! Nodal ordering is interleaved `[ux, uy, uz]` per node. Tetra nodes
//! are `(p0, p1, p2, p3)`; wedge nodes are the bottom triangle `(0, 1,
//! 2)` at `zeta = -1` and the top triangle `(3, 4, 5)` at `zeta = +1`.

// Rust guideline compliant 2026-02-21

use super::super::plasticity::update::StressUpdateModel;
use sonium_compute::explicit_hourglass::{
    BULK_VISCOSITY_LINEAR_DEFAULT, BULK_VISCOSITY_QUADRATIC_DEFAULT, bulk_viscosity_q,
};

/// Error of the tetra/wedge explicit path.
#[derive(Debug, Clone)]
pub struct TetPentaError {
    /// Human-readable failure description.
    pub message: String,
}

impl TetPentaError {
    /// Builds an error from a message.
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl std::fmt::Display for TetPentaError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "tetra/penta explicit error: {}", self.message)
    }
}

impl std::error::Error for TetPentaError {}

/// Explicit-step material parameters shared by tet and wedge.
#[derive(Debug, Clone, Copy)]
pub struct TetPentaParams {
    /// Mass density in kg/m^3.
    pub density: f64,
    /// Sound speed in m/s.
    pub sound_speed: f64,
    /// Quadratic shock-viscosity coefficient (default 1.10).
    pub bulk_c0: f64,
    /// Linear shock-viscosity coefficient (default 0.05).
    pub bulk_c1: f64,
}

impl TetPentaParams {
    /// Builds parameters from density and sound speed with manual defaults.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use sonium_fem::physics::solid::nonlinear_solid::TetPentaParams;
    ///
    /// let params = TetPentaParams::new(7800.0, 4500.0);
    /// assert!((params.bulk_c0 - 1.10).abs() < 1e-15);
    /// ```
    ///
    /// # Panics
    ///
    /// Panics for negative density, sound speed, or viscosity coefficients.
    #[must_use]
    pub fn new(density: f64, sound_speed: f64) -> Self {
        assert!(
            density.is_finite() && density >= 0.0,
            "density must be finite non-negative"
        );
        assert!(
            sound_speed.is_finite() && sound_speed >= 0.0,
            "sound speed must be finite non-negative"
        );
        Self {
            density,
            sound_speed,
            bulk_c0: BULK_VISCOSITY_QUADRATIC_DEFAULT,
            bulk_c1: BULK_VISCOSITY_LINEAR_DEFAULT,
        }
    }
}

/// Constant-strain tetrahedron geometry with its exact B-matrix.
#[derive(Debug, Clone)]
pub struct Tet4Geom {
    /// Node coordinates.
    pub coords: [[f64; 3]; 4],
    /// Element volume in m^3.
    pub volume: f64,
    /// Strain-displacement matrix, 6x12 row-major in C1 Voigt order.
    pub b_matrix: [f64; 72],
    /// Characteristic length `V / A_max` in m.
    pub char_length: f64,
}

impl Tet4Geom {
    /// Builds tetra geometry from four node coordinates.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use sonium_fem::physics::solid::nonlinear_solid::Tet4Geom;
    ///
    /// let geom = Tet4Geom::new([
    ///     [0.0, 0.0, 0.0],
    ///     [1.0, 0.0, 0.0],
    ///     [0.0, 1.0, 0.0],
    ///     [0.0, 0.0, 1.0],
    /// ])
    /// .unwrap();
    /// assert!((geom.volume - 1.0 / 6.0).abs() < 1e-15);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`TetPentaError`] for non-finite coordinates or a
    /// degenerate (non-positive volume) tetrahedron.
    pub fn new(coords: [[f64; 3]; 4]) -> Result<Self, TetPentaError> {
        if !coords.iter().flatten().all(|v| v.is_finite()) {
            return Err(TetPentaError::new("tetra coordinates must be finite"));
        }
        // Jacobian columns from node 0; volume is det / 6.
        let mut j = [0.0; 9];
        for col in 0..3 {
            for row in 0..3 {
                j[row * 3 + col] = coords[col + 1][row] - coords[0][row];
            }
        }
        let det = det3(&j);
        if det.is_nan() || det <= 0.0 {
            return Err(TetPentaError::new("tetrahedron must have positive volume"));
        }
        let volume = det / 6.0;
        let inv = invert3(&j, det);
        // Gradients: dN_0 = -(row sums), dN_i = row i - 1 of J^-1.
        let mut grad = [[0.0; 3]; 4];
        for (slot, row) in grad.iter_mut().skip(1).enumerate() {
            *row = [inv[slot * 3], inv[slot * 3 + 1], inv[slot * 3 + 2]];
        }
        grad[0] = std::array::from_fn(|j| -(grad[1][j] + grad[2][j] + grad[3][j]));
        let mut b_matrix = [0.0; 72];
        for (node, g) in grad.iter().enumerate() {
            // Normal rows.
            b_matrix[node * 3] = g[0];
            b_matrix[12 + node * 3 + 1] = g[1];
            b_matrix[24 + node * 3 + 2] = g[2];
            // Engineering shear rows: xy, yz, zx in C1 order.
            b_matrix[36 + node * 3] = g[1];
            b_matrix[36 + node * 3 + 1] = g[0];
            b_matrix[48 + node * 3 + 1] = g[2];
            b_matrix[48 + node * 3 + 2] = g[1];
            b_matrix[60 + node * 3] = g[2];
            b_matrix[60 + node * 3 + 2] = g[0];
        }
        // Largest face area sets the conservative length V / A_max.
        let faces = [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]];
        let mut max_area: f64 = 0.0;
        for face in faces {
            max_area = max_area.max(tri_area(
                &coords[face[0]],
                &coords[face[1]],
                &coords[face[2]],
            ));
        }
        Ok(Self {
            coords,
            volume,
            b_matrix,
            char_length: volume / max_area,
        })
    }

    /// Lumped nodal masses in kg, one quarter of the total each.
    #[must_use]
    pub fn lumped_masses(&self, density: f64) -> [f64; 4] {
        let nodal = density * self.volume / 4.0;
        [nodal; 4]
    }

    /// Critical explicit step `L / c` in s.
    #[must_use]
    pub fn critical_step(&self, sound_speed: f64) -> f64 {
        self.char_length / sound_speed
    }
}

/// History state of one tetrahedron (single material point).
#[derive(Debug, Clone)]
pub struct Tet4State<S> {
    /// Material state at the centroid point.
    pub point: S,
}

impl<S> Tet4State<S> {
    /// Builds tetra state from a material-point state.
    #[must_use]
    pub fn new(point: S) -> Self {
        Self { point }
    }
}

/// Internal-force evaluation of one tetrahedron.
#[derive(Debug, Clone)]
pub struct TetForceOut {
    /// Nodal internal forces in N, interleaved per node.
    pub forces: [f64; 12],
    /// Shock viscosity pressure in Pa.
    pub bulk_q: f64,
    /// Equivalent plastic strain increment of the step.
    pub dep: f64,
    /// True when the point yielded.
    pub yielding: bool,
}

/// Explicit internal forces of one TETRA4 element.
///
/// Advances the centroid C1 material by the strain increment from the
/// nodal velocities over `dt`, commits it, and returns `V B^T (sigma +
/// q I)`. Rate-dependent materials evaluate at the estimated rate
/// `|d_eps| / dt`.
///
/// # Errors
///
/// Returns [`TetPentaError`] for non-positive `dt` or non-finite input.
pub fn tet4_forces<M>(
    geom: &Tet4Geom,
    state: &mut Tet4State<M::State>,
    material: &M,
    velocities: &[f64; 12],
    dt: f64,
    params: &TetPentaParams,
) -> Result<TetForceOut, TetPentaError>
where
    M: StressUpdateModel<StrainIncrement = [f64; 6]>,
{
    if dt.is_nan() || dt <= 0.0 {
        return Err(TetPentaError::new("explicit step dt must be positive"));
    }
    if !velocities.iter().all(|v| v.is_finite()) {
        return Err(TetPentaError::new("tetra velocities must be finite"));
    }
    let mut d_eps = [0.0; 6];
    for (slot, brow) in d_eps.iter_mut().zip(geom.b_matrix.chunks(12)) {
        let mut sum = 0.0;
        for (b, v) in brow.iter().zip(velocities.iter()) {
            sum += b * v * dt;
        }
        *slot = sum;
    }
    let vol_rate = (d_eps[0] + d_eps[1] + d_eps[2]) / dt;
    let bulk_q = bulk_viscosity_q(
        params.density,
        params.sound_speed,
        geom.char_length,
        params.bulk_c0,
        params.bulk_c1,
        vol_rate,
    );
    let update = if material.is_rate_dependent() {
        material.update_with_rate(&state.point, &d_eps, dt)
    } else {
        material.update(&state.point, &d_eps)
    };
    material.commit(&mut state.point, &update);
    let stresses: [f64; 6] =
        std::array::from_fn(|row| update.stress[row] + if row < 3 { bulk_q } else { 0.0 });
    let mut forces = [0.0; 12];
    for (brow, s) in geom.b_matrix.chunks(12).zip(stresses.iter()) {
        for (f, b) in forces.iter_mut().zip(brow.iter()) {
            *f += b * s;
        }
    }
    for f in forces.iter_mut() {
        *f *= geom.volume;
    }
    Ok(TetForceOut {
        forces,
        bulk_q,
        dep: update.delta_equiv_plastic_strain,
        yielding: update.yielding,
    })
}

/// Six-node wedge geometry (PENTA6).
///
/// Bottom triangle `(0, 1, 2)` at `zeta = -1`, top triangle `(3, 4, 5)`
/// at `zeta = +1`. Shape functions are triangle barycentric functions
/// times `(1 -/+ zeta) / 2`.
#[derive(Debug, Clone)]
pub struct Penta6Geom {
    /// Node coordinates.
    pub coords: [[f64; 3]; 6],
    /// Element volume in m^3 from full quadrature.
    pub volume: f64,
    /// Characteristic length `V / A_max` in m.
    pub char_length: f64,
}

/// One wedge quadrature point in natural coordinates.
#[derive(Debug, Clone, Copy)]
pub struct PentaPoint {
    /// Triangle barycentric coordinates `(r, s, t)`.
    pub bary: [f64; 3],
    /// Through-thickness coordinate in `[-1, 1]`.
    pub zeta: f64,
    /// Quadrature weight over the reference wedge (volume 1).
    pub weight: f64,
}

/// Full 6-point wedge quadrature (triangle x zeta).
///
/// Combines the 3-point degree-2 triangle rule (barycentric
/// permutations of `(2/3, 1/6, 1/6)`, weight `1/6` each... note the
/// triangle rule sums to the reference area `1/2`) with the 2-point
/// Gauss rule in zeta (weight 1 each), so weights sum to the reference
/// wedge volume 1.
///
/// # Examples
///
/// ```rust
/// use sonium_fem::physics::solid::nonlinear_solid::penta_quadrature;
///
/// let rule = penta_quadrature();
/// let total: f64 = rule.iter().map(|p| p.weight).sum();
/// assert!((total - 1.0).abs() < 1e-15);
/// ```
#[must_use]
pub fn penta_quadrature() -> [PentaPoint; 6] {
    let sixth = 1.0 / 6.0;
    let third = 2.0 / 3.0;
    let gauss = 1.0 / 3.0_f64.sqrt();
    let triangles = [
        [third, sixth, sixth],
        [sixth, third, sixth],
        [sixth, sixth, third],
    ];
    let mut points = [PentaPoint {
        bary: [0.0; 3],
        zeta: 0.0,
        weight: 0.0,
    }; 6];
    let mut slot = 0;
    for zeta in [-gauss, gauss] {
        for bary in triangles {
            // Triangle weight 1/6 (rule sums to area 1/2) times zeta
            // Gauss weight 1: total reference volume 1.
            points[slot] = PentaPoint {
                bary,
                zeta,
                weight: sixth,
            };
            slot += 1;
        }
    }
    points
}

/// Wedge shape functions at a natural point.
#[must_use]
pub fn penta_shape(bary: &[f64; 3], zeta: f64) -> [f64; 6] {
    let bottom = 0.5 * (1.0 - zeta);
    let top = 0.5 * (1.0 + zeta);
    [
        bary[0] * bottom,
        bary[1] * bottom,
        bary[2] * bottom,
        bary[0] * top,
        bary[1] * top,
        bary[2] * top,
    ]
}

impl Penta6Geom {
    /// Builds wedge geometry from six node coordinates.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use sonium_fem::physics::solid::nonlinear_solid::Penta6Geom;
    ///
    /// let geom = Penta6Geom::new([
    ///     [0.0, 0.0, 0.0],
    ///     [1.0, 0.0, 0.0],
    ///     [0.0, 1.0, 0.0],
    ///     [0.0, 0.0, 1.0],
    ///     [1.0, 0.0, 1.0],
    ///     [0.0, 1.0, 1.0],
    /// ])
    /// .unwrap();
    /// assert!((geom.volume - 0.5).abs() < 1e-14);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`TetPentaError`] for non-finite coordinates or a
    /// degenerate (non-positive Jacobian) wedge.
    pub fn new(coords: [[f64; 3]; 6]) -> Result<Self, TetPentaError> {
        if !coords.iter().flatten().all(|v| v.is_finite()) {
            return Err(TetPentaError::new("wedge coordinates must be finite"));
        }
        let mut volume = 0.0;
        for point in penta_quadrature() {
            let (_, det) = penta_map(&coords, &point)?;
            volume += point.weight * det;
        }
        if volume.is_nan() || volume <= 0.0 {
            return Err(TetPentaError::new("wedge must have positive volume"));
        }
        // Faces: 2 triangles + 3 quads (each split into 2 triangles).
        let tris = [[0, 1, 2], [3, 4, 5]];
        let quads = [[0, 1, 4, 3], [1, 2, 5, 4], [2, 0, 3, 5]];
        let mut max_area: f64 = 0.0;
        for tri in tris {
            max_area = max_area.max(tri_area(&coords[tri[0]], &coords[tri[1]], &coords[tri[2]]));
        }
        for quad in quads {
            let area = tri_area(&coords[quad[0]], &coords[quad[1]], &coords[quad[2]])
                + tri_area(&coords[quad[0]], &coords[quad[2]], &coords[quad[3]]);
            max_area = max_area.max(area);
        }
        Ok(Self {
            coords,
            volume,
            char_length: volume / max_area,
        })
    }

    /// Lumped nodal masses in kg, one sixth of the total each.
    #[must_use]
    pub fn lumped_masses(&self, density: f64) -> [f64; 6] {
        let nodal = density * self.volume / 6.0;
        [nodal; 6]
    }

    /// Critical explicit step `L / c` in s.
    #[must_use]
    pub fn critical_step(&self, sound_speed: f64) -> f64 {
        self.char_length / sound_speed
    }
}

/// History state of one wedge (six material points).
#[derive(Debug, Clone)]
pub struct Penta6State<S> {
    /// Material states at the six quadrature points.
    pub points: [S; 6],
}

impl<S> Penta6State<S> {
    /// Builds wedge state from six material-point states.
    #[must_use]
    pub fn new(points: [S; 6]) -> Self {
        Self { points }
    }
}

/// Internal-force evaluation of one wedge.
#[derive(Debug, Clone)]
pub struct PentaForceOut {
    /// Nodal internal forces in N, interleaved per node.
    pub forces: [f64; 18],
    /// Peak shock viscosity pressure over the points in Pa.
    pub bulk_q: f64,
    /// Peak equivalent plastic increment over the points.
    pub dep: f64,
    /// True when any point yielded.
    pub yielding: bool,
}

/// Explicit internal forces of one PENTA6 element.
///
/// Advances each of the six C1 material points by its strain increment
/// and returns the quadrature sum of `w B^T (sigma + q I)`.
/// Rate-dependent materials evaluate at `|d_eps| / dt` per point.
///
/// # Errors
///
/// Returns [`TetPentaError`] for non-positive `dt`, non-finite input,
/// or a degenerate quadrature map.
pub fn penta6_forces<M>(
    geom: &Penta6Geom,
    state: &mut Penta6State<M::State>,
    material: &M,
    velocities: &[f64; 18],
    dt: f64,
    params: &TetPentaParams,
) -> Result<PentaForceOut, TetPentaError>
where
    M: StressUpdateModel<StrainIncrement = [f64; 6]>,
{
    if dt.is_nan() || dt <= 0.0 {
        return Err(TetPentaError::new("explicit step dt must be positive"));
    }
    if !velocities.iter().all(|v| v.is_finite()) {
        return Err(TetPentaError::new("wedge velocities must be finite"));
    }
    let mut forces = [0.0; 18];
    let mut bulk_q: f64 = 0.0;
    let mut dep: f64 = 0.0;
    let mut yielding = false;
    for (slot, point) in penta_quadrature().iter().enumerate() {
        let (b_matrix, det) = penta_map(&geom.coords, point)?;
        let mut d_eps = [0.0; 6];
        for (slot, brow) in d_eps.iter_mut().zip(b_matrix.chunks(18)) {
            let mut sum = 0.0;
            for (b, v) in brow.iter().zip(velocities.iter()) {
                sum += b * v * dt;
            }
            *slot = sum;
        }
        let vol_rate = (d_eps[0] + d_eps[1] + d_eps[2]) / dt;
        let q = bulk_viscosity_q(
            params.density,
            params.sound_speed,
            geom.char_length,
            params.bulk_c0,
            params.bulk_c1,
            vol_rate,
        );
        let update = if material.is_rate_dependent() {
            material.update_with_rate(&state.points[slot], &d_eps, dt)
        } else {
            material.update(&state.points[slot], &d_eps)
        };
        material.commit(&mut state.points[slot], &update);
        let weight = point.weight * det;
        let stresses: [f64; 6] =
            std::array::from_fn(|row| update.stress[row] + if row < 3 { q } else { 0.0 });
        for (brow, s) in b_matrix.chunks(18).zip(stresses.iter()) {
            for (f, b) in forces.iter_mut().zip(brow.iter()) {
                *f += b * s * weight;
            }
        }
        bulk_q = bulk_q.max(q);
        dep = dep.max(update.delta_equiv_plastic_strain);
        yielding |= update.yielding;
    }
    Ok(PentaForceOut {
        forces,
        bulk_q,
        dep,
        yielding,
    })
}

/// B-matrix (6x18) and Jacobian determinant at a wedge point.
///
/// Differentiates in `(r, s, zeta)`: the triangle directions use
/// `dL = [(-1, -1), (1, 0), (0, 1)]` rows, zeta uses `(top -
/// bottom) / 2 ... ` exactly `(shape_top - shape_bottom) / 1`
/// scaled by the `1/2` factors (bottom derivative `-L/2`, top `+L/2`).
fn penta_map(
    coords: &[[f64; 3]; 6],
    point: &PentaPoint,
) -> Result<([f64; 108], f64), TetPentaError> {
    // Natural derivatives dN/d(r, s, zeta).
    let bottom = 0.5 * (1.0 - point.zeta);
    let top = 0.5 * (1.0 + point.zeta);
    // Triangle gradients: L0 = 1 - r - s, L1 = r, L2 = s.
    let tri_grad = [[-1.0, -1.0], [1.0, 0.0], [0.0, 1.0]];
    let mut deriv = [[0.0; 3]; 6];
    for i in 0..3 {
        deriv[i][0] = tri_grad[i][0] * bottom;
        deriv[i][1] = tri_grad[i][1] * bottom;
        deriv[i][2] = -0.5 * point.bary[i];
        deriv[i + 3][0] = tri_grad[i][0] * top;
        deriv[i + 3][1] = tri_grad[i][1] * top;
        deriv[i + 3][2] = 0.5 * point.bary[i];
    }
    let mut j = [0.0; 9];
    for (node, d) in deriv.iter().enumerate() {
        for row in 0..3 {
            for col in 0..3 {
                j[row * 3 + col] += coords[node][row] * d[col];
            }
        }
    }
    let det = det3(&j);
    if det.is_nan() || det <= 0.0 {
        return Err(TetPentaError::new("wedge quadrature map is degenerate"));
    }
    let inv = invert3(&j, det);
    let mut grad = [[0.0; 3]; 6];
    for (node, d) in deriv.iter().enumerate() {
        for i in 0..3 {
            grad[node][i] = inv[i] * d[0] + inv[3 + i] * d[1] + inv[6 + i] * d[2];
        }
    }
    let mut b = [0.0; 108];
    for (node, g) in grad.iter().enumerate() {
        b[node * 3] = g[0];
        b[18 + node * 3 + 1] = g[1];
        b[36 + node * 3 + 2] = g[2];
        b[54 + node * 3] = g[1];
        b[54 + node * 3 + 1] = g[0];
        b[72 + node * 3 + 1] = g[2];
        b[72 + node * 3 + 2] = g[1];
        b[90 + node * 3] = g[2];
        b[90 + node * 3 + 2] = g[0];
    }
    Ok((b, det))
}

/// Area of a triangle from three points.
fn tri_area(p0: &[f64; 3], p1: &[f64; 3], p2: &[f64; 3]) -> f64 {
    let u = [p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]];
    let v = [p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]];
    let cross = [
        u[1] * v[2] - u[2] * v[1],
        u[2] * v[0] - u[0] * v[2],
        u[0] * v[1] - u[1] * v[0],
    ];
    0.5 * (cross[0] * cross[0] + cross[1] * cross[1] + cross[2] * cross[2]).sqrt()
}

/// Determinant of a row-major 3x3 matrix.
fn det3(m: &[f64; 9]) -> f64 {
    m[0] * (m[4] * m[8] - m[5] * m[7]) - m[1] * (m[3] * m[8] - m[5] * m[6])
        + m[2] * (m[3] * m[7] - m[4] * m[6])
}

/// Inverse of a row-major 3x3 matrix with known determinant.
fn invert3(m: &[f64; 9], det: f64) -> [f64; 9] {
    let inv_det = 1.0 / det;
    [
        (m[4] * m[8] - m[5] * m[7]) * inv_det,
        (m[2] * m[7] - m[1] * m[8]) * inv_det,
        (m[1] * m[5] - m[2] * m[4]) * inv_det,
        (m[5] * m[6] - m[3] * m[8]) * inv_det,
        (m[0] * m[8] - m[2] * m[6]) * inv_det,
        (m[2] * m[3] - m[0] * m[5]) * inv_det,
        (m[3] * m[7] - m[4] * m[6]) * inv_det,
        (m[1] * m[6] - m[0] * m[7]) * inv_det,
        (m[0] * m[4] - m[1] * m[3]) * inv_det,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::physics::solid::plasticity::crash::{JohnsonCookMaterial, JohnsonCookState};
    use crate::physics::solid::plasticity::{HardeningLaw, J2Material, PlasticState};

    fn elastic_j2() -> J2Material {
        J2Material::new(200.0e9, 0.3, 1.0e12, HardeningLaw::Perfect)
    }

    #[test]
    fn reference_tet_b_matrix_is_exact() {
        let geom = Tet4Geom::new([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ])
        .unwrap();
        assert!((geom.volume - 1.0 / 6.0).abs() < 1e-16);
        // Gradients: N0 = (-1,-1,-1), N1 = (1,0,0), N2 = (0,1,0), N3 = (0,0,1).
        assert_eq!(geom.b_matrix[0], -1.0);
        assert_eq!(geom.b_matrix[3], 1.0);
        assert_eq!(geom.b_matrix[12 + 7], 1.0);
        assert_eq!(geom.b_matrix[24 + 11], 1.0);
        // Shear row xy at node 1: [dNy, dNx] = [0, 1].
        assert_eq!(geom.b_matrix[36 + 3], 0.0);
        assert_eq!(geom.b_matrix[36 + 4], 1.0);
    }

    #[test]
    fn tet_patch_gives_constant_stress() {
        let geom = Tet4Geom::new([
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ])
        .unwrap();
        let material = elastic_j2();
        let mut state = Tet4State::new(PlasticState::new_3d());
        let params = TetPentaParams::new(7800.0, 4500.0);
        // Uniaxial stretch rate: exx_dot = 1/s over dt = 1e-3.
        let mut velocities = [0.0; 12];
        for node in 0..4 {
            velocities[node * 3] = geom.coords[node][0];
        }
        let out = tet4_forces(&geom, &mut state, &material, &velocities, 1.0e-3, &params).unwrap();
        assert!(!out.yielding);
        assert_eq!(out.dep, 0.0);
        // Elastic uniaxial-strain stress: sxx = (K + 4G/3) e, e = 1e-3.
        let g = 200.0e9 / 2.6;
        let k = 200.0e9 / 1.2;
        let expected = (k + 4.0 * g / 3.0) * 1.0e-3;
        // Recovered from force equilibrium: sum x-forces vanish.
        let fx: f64 = [0, 3, 6, 9].iter().map(|i| out.forces[*i]).sum();
        assert!(fx.abs() < 1e-6 * expected * geom.volume);
        // Point stress matches the closed form (bulk q is zero: d > 0).
        assert_eq!(out.bulk_q, 0.0);
        assert!((state.point.stress[0] - expected).abs() < 1e-3 * expected);
        assert!(out.forces.iter().any(|f| f.abs() > 0.0));
    }

    #[test]
    fn tet_rigid_translation_gives_zero_forces() {
        let geom = Tet4Geom::new([
            [0.1, 0.2, 0.0],
            [1.1, 0.1, 0.2],
            [0.2, 1.3, 0.1],
            [0.0, 0.2, 1.1],
        ])
        .unwrap();
        let material = elastic_j2();
        let mut state = Tet4State::new(PlasticState::new_3d());
        let params = TetPentaParams::new(7800.0, 4500.0);
        let velocities = [
            2.0, -1.0, 0.5, 2.0, -1.0, 0.5, 2.0, -1.0, 0.5, 2.0, -1.0, 0.5,
        ];
        let out = tet4_forces(&geom, &mut state, &material, &velocities, 1.0e-3, &params).unwrap();
        // Rounding floor: E amplifies 1e-16 strain cancellation to ~1e-7.
        assert!(out.forces.iter().all(|f| f.abs() < 1e-6));
    }

    #[test]
    fn tet_bulk_viscosity_acts_in_compression() {
        let geom = Tet4Geom::new([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ])
        .unwrap();
        let material = elastic_j2();
        let params = TetPentaParams::new(7800.0, 4500.0);
        // Uniform compression rate toward the origin.
        let mut velocities = [0.0; 12];
        for node in 0..4 {
            velocities[node * 3] = -geom.coords[node][0];
            velocities[node * 3 + 1] = -geom.coords[node][1];
            velocities[node * 3 + 2] = -geom.coords[node][2];
        }
        let mut state = Tet4State::new(PlasticState::new_3d());
        let out = tet4_forces(&geom, &mut state, &material, &velocities, 1.0e-3, &params).unwrap();
        assert!(out.bulk_q > 0.0);
    }

    #[test]
    fn tet_rate_path_uses_johnson_cook() {
        let geom = Tet4Geom::new([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ])
        .unwrap();
        let material = JohnsonCookMaterial::c4340_steel();
        assert!(material.is_rate_dependent());
        let params = TetPentaParams::new(7800.0, 4500.0);
        let mut velocities = [0.0; 12];
        for node in 0..4 {
            velocities[node * 3] = geom.coords[node][0] * 100.0;
        }
        let mut state = Tet4State::new(JohnsonCookState::new(293.0));
        let out = tet4_forces(&geom, &mut state, &material, &velocities, 1.0e-3, &params).unwrap();
        assert!(out.forces.iter().any(|f| f.abs() > 0.0));
        assert!(state.point.equiv_plastic_strain >= 0.0);
    }

    #[test]
    fn tet_rejects_degenerate_input() {
        let flat = Tet4Geom::new([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
        ]);
        assert!(flat.is_err());
        let geom = Tet4Geom::new([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ])
        .unwrap();
        let material = elastic_j2();
        let mut state = Tet4State::new(PlasticState::new_3d());
        let params = TetPentaParams::new(7800.0, 4500.0);
        assert!(tet4_forces(&geom, &mut state, &material, &[0.0; 12], 0.0, &params).is_err());
    }

    #[test]
    fn penta_unit_wedge_volume_and_mass() {
        let geom = Penta6Geom::new([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 2.0],
            [1.0, 0.0, 2.0],
            [0.0, 1.0, 2.0],
        ])
        .unwrap();
        // Triangle area 1/2 times height 2.
        assert!((geom.volume - 1.0).abs() < 1e-14);
        let masses = geom.lumped_masses(7800.0);
        let total: f64 = masses.iter().sum();
        assert!((total - 7800.0).abs() < 1e-9);
        assert!(geom.char_length > 0.0);
        assert!(geom.critical_step(4500.0) > 0.0);
    }

    #[test]
    fn penta_patch_is_self_equilibrated() {
        let geom = Penta6Geom::new([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
        ])
        .unwrap();
        let material = elastic_j2();
        let mut state = Penta6State::new(std::array::from_fn(|_| PlasticState::new_3d()));
        let params = TetPentaParams::new(7800.0, 4500.0);
        // Symmetric stretch keeps stress symmetric: forces and moments balance.
        let mut velocities = [0.0; 18];
        for node in 0..6 {
            let [x, y, z] = geom.coords[node];
            velocities[node * 3] = 0.02 * x + 0.01 * y + 0.005 * z;
            velocities[node * 3 + 1] = 0.01 * x + 0.03 * y - 0.004 * z;
            velocities[node * 3 + 2] = 0.005 * x - 0.004 * y + 0.025 * z;
        }
        let out =
            penta6_forces(&geom, &mut state, &material, &velocities, 1.0e-3, &params).unwrap();
        let mut force = [0.0; 3];
        let mut moment = [0.0; 3];
        for node in 0..6 {
            let x = geom.coords[node];
            let f = [
                out.forces[node * 3],
                out.forces[node * 3 + 1],
                out.forces[node * 3 + 2],
            ];
            for i in 0..3 {
                force[i] += f[i];
            }
            moment[0] += x[1] * f[2] - x[2] * f[1];
            moment[1] += x[2] * f[0] - x[0] * f[2];
            moment[2] += x[0] * f[1] - x[1] * f[0];
        }
        let scale: f64 = out.forces.iter().map(|f| f.abs()).sum::<f64>().max(1.0e-30);
        for i in 0..3 {
            assert!(force[i].abs() / scale < 1.0e-12);
            assert!(moment[i].abs() / scale < 1.0e-9);
        }
    }

    #[test]
    fn penta_rigid_motion_gives_zero_forces() {
        let geom = Penta6Geom::new([
            [0.1, 0.0, 0.2],
            [1.2, 0.1, 0.0],
            [0.0, 1.1, 0.3],
            [0.2, 0.1, 1.0],
            [1.0, 0.0, 1.2],
            [0.1, 0.9, 1.1],
        ])
        .unwrap();
        let material = elastic_j2();
        let mut state = Penta6State::new(std::array::from_fn(|_| PlasticState::new_3d()));
        let params = TetPentaParams::new(7800.0, 4500.0);
        let mut velocities = [0.0; 18];
        for node in 0..6 {
            velocities[node * 3] = 1.5;
            velocities[node * 3 + 1] = -2.0;
            velocities[node * 3 + 2] = 0.25;
        }
        let out =
            penta6_forces(&geom, &mut state, &material, &velocities, 1.0e-3, &params).unwrap();
        let scale = 200.0e9 * geom.volume * 1.0e-3;
        assert!(out.forces.iter().all(|f| f.abs() / scale < 1.0e-12));
    }

    #[test]
    fn penta_shape_partitions_unity() {
        for point in penta_quadrature() {
            let n = penta_shape(&point.bary, point.zeta);
            let sum: f64 = n.iter().sum();
            assert!((sum - 1.0).abs() < 1e-15);
        }
    }
}
