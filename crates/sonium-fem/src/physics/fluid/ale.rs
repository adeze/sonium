//! Moving-mesh ALE kernels on structured quad grids.
//!
//! Self-contained arbitrary Lagrangian-Eulerian (ALE) building blocks:
//! a movable structured 2D quad grid, an unsplit ALE finite-volume
//! update with exact swept-volume geometric conservation, a
//! conservative donor-cell remap, mesh smoothing, a periodic 1D
//! advection kernel (Eulerian limit), and a 1D elastic bar-impact
//! reference (Lagrangian limit). Only material-level helpers are
//! reused from `super::compressible_flow` (`ConservativeState`,
//! `IdealGas`); no existing file is edited.
//!
//! # Decision spike verdict: BUILD
//!
//! The ALE2D/bar2_v41 deck (vendor miniqa case: 2D copper-bar impact
//! at 227 m/s, `/MAT/LAW3` elastoplastic, `/ALE/GRID/DONEA` grid,
//! `/QUAD` mesh of 306 nodes, `/ALE/BCS` walls) needs quad-mesh
//! topology, mesh-motion-aware fluxes, a geometric conservation law
//! (GCL), remap/advection transport, and elastoplastic response. The
//! existing `compressible_flow` DG core offers an Eulerian P1-DG gas
//! solver on a fixed simplex mesh (`CompressibleDgConfig`,
//! `CompressibleDgState`, `CompressibleDgOperator`): it rejects quad
//! meshes (`validate_simplex_mesh`), borrows its mesh immutably with
//! no mesh-velocity hooks in `spatial_operator`, and has no remap,
//! DONEA smoothing, or solid constitutive laws. A thin WRAP driver
//! that rebuilds the Eulerian operator on a moved mesh would be
//! re-meshing, not ALE: without relative-velocity fluxes and GCL
//! metrics it cannot preserve free streams or conserve on moving
//! volumes. Hence BUILD: new mesh-motion plus remap/advection here,
//! reusing only the public material API. No blocking limitation in
//! the DG core requires a coordinator ruling.
//!
//! # Equations
//!
//! The unsplit update discretizes the ALE conservation law on a cell
//! of volume `V(t)` with mesh velocity `w` (Donea et al., ALE
//! methods; Radioss Theory Manual ALE chapter):
//!
//! ```text
//! d/dt (V U) + sum_facets (F(U) - w_n U) l = 0,
//! ```
//!
//! with the Rusanov numerical flux evaluated at the relative normal
//! velocity `(u - w) . n`. Volumes follow from swept facets, so the
//! discrete GCL `V_new - V_old = sum w_n l dt` holds to roundoff and
//! uniform states are preserved exactly on arbitrary mesh motion.
//! The split remap moves donor-cell means between two grids of one
//! topology through the same swept volumes.
//!
//! # Examples
//!
//! ```rust
//! use sonium_fem::physics::fluid::ale::{BarImpact, QuadGrid};
//!
//! let grid = QuadGrid::rectangle(4, 2, 0.0, 1.0, 0.0, 0.5).unwrap();
//! assert_eq!(grid.num_cells(), 8);
//! let bar = BarImpact::copper_bar2();
//! assert!(bar.contact_stress() > 7.0e9);
//! ```

// Rust guideline compliant 2026-02-21

use std::error::Error;
use std::fmt::{Display, Formatter};

use super::compressible_flow::{ConservativeState, IdealGas};

/// ALE kernel failure modes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AleError {
    /// Grid dimensions or node layout are inconsistent.
    InvalidGrid,
    /// State length does not match the grid cell count.
    StateSizeMismatch,
    /// A cell volume is not strictly positive.
    NonPositiveVolume,
    /// A state has non-positive density or pressure.
    NonPhysicalState,
    /// A scalar parameter is out of its valid range.
    InvalidParameter,
}

impl Display for AleError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidGrid => write!(f, "invalid ALE grid layout"),
            Self::StateSizeMismatch => write!(f, "ALE state/grid size mismatch"),
            Self::NonPositiveVolume => write!(f, "non-positive ALE cell volume"),
            Self::NonPhysicalState => write!(f, "non-physical ALE state"),
            Self::InvalidParameter => write!(f, "invalid ALE parameter"),
        }
    }
}

impl Error for AleError {}

/// Structured movable 2D quad grid.
#[derive(Debug, Clone, PartialEq)]
pub struct QuadGrid {
    nx: usize,
    ny: usize,
    nodes: Vec<[f64; 2]>,
}

impl QuadGrid {
    /// Builds a uniform rectangle grid.
    ///
    /// Nodes are row-major with `nx + 1` columns; `x0 < x1` and
    /// `y0 < y1` with at least one cell per direction required.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use sonium_fem::physics::fluid::ale::QuadGrid;
    ///
    /// let grid = QuadGrid::rectangle(2, 3, 0.0, 2.0, 0.0, 3.0).unwrap();
    /// assert_eq!(grid.num_cells(), 6);
    /// assert_eq!(grid.num_nodes(), 12);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns `AleError::InvalidGrid` for empty dimensions, inverted
    /// bounds, or non-finite coordinates.
    pub fn rectangle(
        nx: usize,
        ny: usize,
        x0: f64,
        x1: f64,
        y0: f64,
        y1: f64,
    ) -> Result<Self, AleError> {
        if nx == 0
            || ny == 0
            || !x0.is_finite()
            || !x1.is_finite()
            || !y0.is_finite()
            || !y1.is_finite()
            || x1 <= x0
            || y1 <= y0
        {
            return Err(AleError::InvalidGrid);
        }
        let mut nodes = Vec::with_capacity((nx + 1) * (ny + 1));
        for j in 0..=ny {
            for i in 0..=nx {
                nodes.push([
                    x0 + (x1 - x0) * i as f64 / nx as f64,
                    y0 + (y1 - y0) * j as f64 / ny as f64,
                ]);
            }
        }
        Ok(Self { nx, ny, nodes })
    }

    /// Returns the cell count in x.
    pub fn nx(&self) -> usize {
        self.nx
    }

    /// Returns the cell count in y.
    pub fn ny(&self) -> usize {
        self.ny
    }

    /// Returns the total cell count.
    pub fn num_cells(&self) -> usize {
        self.nx * self.ny
    }

    /// Returns the total node count.
    pub fn num_nodes(&self) -> usize {
        self.nodes.len()
    }

    /// Returns node `(i, j)` coordinates.
    ///
    /// # Panics
    ///
    /// Panics when the indices exceed the grid; callers pass structured
    /// loop indices that always satisfy the bounds by construction.
    pub fn node(&self, i: usize, j: usize) -> [f64; 2] {
        self.nodes[self.node_index(i, j)]
    }

    /// Sets node `(i, j)` coordinates.
    ///
    /// # Panics
    ///
    /// Panics when the indices exceed the grid; callers pass structured
    /// loop indices that always satisfy the bounds by construction.
    pub fn set_node(&mut self, i: usize, j: usize, point: [f64; 2]) {
        let index = self.node_index(i, j);
        self.nodes[index] = point;
    }

    /// Returns cell node indices (SW, SE, NE, NW).
    ///
    /// # Panics
    ///
    /// Panics when `cell` exceeds the cell count; callers iterate
    /// `0..num_cells`, which always satisfies the bound.
    pub fn cell_nodes(&self, cell: usize) -> [usize; 4] {
        let i = cell % self.nx;
        let j = cell / self.nx;
        assert!(j < self.ny, "cell index exceeds grid rows");
        [
            self.node_index(i, j),
            self.node_index(i + 1, j),
            self.node_index(i + 1, j + 1),
            self.node_index(i, j + 1),
        ]
    }

    /// Returns the signed quad area of a cell.
    ///
    /// Positive for counter-clockwise quads; ALE steps reject
    /// non-positive volumes before use.
    ///
    /// # Panics
    ///
    /// Panics when `cell` exceeds the cell count; callers iterate
    /// `0..num_cells`, which always satisfies the bound.
    pub fn cell_volume(&self, cell: usize) -> f64 {
        let corners = self.cell_nodes(cell);
        0.5 * shoelace_sum([
            self.nodes[corners[0]],
            self.nodes[corners[1]],
            self.nodes[corners[2]],
            self.nodes[corners[3]],
        ])
    }

    /// Returns all cell volumes in cell order.
    pub fn volumes(&self) -> Vec<f64> {
        (0..self.num_cells())
            .map(|cell| self.cell_volume(cell))
            .collect()
    }

    /// Returns the vertex-averaged cell centroid.
    ///
    /// # Panics
    ///
    /// Panics when `cell` exceeds the cell count; callers iterate
    /// `0..num_cells`, which always satisfies the bound.
    pub fn cell_centroid(&self, cell: usize) -> [f64; 2] {
        let corners = self.cell_nodes(cell);
        let mut center = [0.0; 2];
        for corner in corners {
            center[0] += self.nodes[corner][0];
            center[1] += self.nodes[corner][1];
        }
        [center[0] / 4.0, center[1] / 4.0]
    }

    /// Rejects grids of differing topology.
    ///
    /// # Errors
    ///
    /// Returns `AleError::InvalidGrid` when dimensions differ.
    pub fn check_topology(&self, other: &Self) -> Result<(), AleError> {
        if self.nx != other.nx || self.ny != other.ny {
            return Err(AleError::InvalidGrid);
        }
        Ok(())
    }

    fn node_index(&self, i: usize, j: usize) -> usize {
        assert!(i <= self.nx, "node column exceeds grid");
        assert!(j <= self.ny, "node row exceeds grid");
        j * (self.nx + 1) + i
    }
}

/// Shoelace edge sum over a closed polygon.
fn shoelace_sum(polygon: [[f64; 2]; 4]) -> f64 {
    let mut sum = 0.0;
    for k in 0..4 {
        let current = polygon[k];
        let next = polygon[(k + 1) % 4];
        sum += current[0] * next[1] - next[0] * current[1];
    }
    sum
}

/// Swept volume gained by the left-side cell.
///
/// The directed facet runs from `a` to `b` (old then new positions);
/// returns the volume gained on its left. Vertical facets pass
/// bottom-to-top (left is the minus cell); horizontal facets pass
/// right-to-left (left is the below minus cell). Negated shoelace
/// area of the space-time quad, exact for piecewise-linear paths.
///
/// # Examples
///
/// ```rust
/// use sonium_fem::physics::fluid::ale::swept_gain;
///
/// // Vertical facet translating +x by 0.5 over unit height.
/// let gain = swept_gain([0.0, 0.0], [0.0, 1.0], [0.5, 0.0], [0.5, 1.0]);
/// assert!((gain - 0.5).abs() < 1.0e-15);
/// ```
pub fn swept_gain(a_old: [f64; 2], b_old: [f64; 2], a_new: [f64; 2], b_new: [f64; 2]) -> f64 {
    -0.5 * shoelace_sum([a_old, b_old, b_new, a_new])
}

/// ALE Rusanov flux with relative normal velocity.
///
/// Evaluates the Eulerian Rusanov flux at `(u - w) . n` and subtracts
/// the mesh transport `w_n U`; identical to `rusanov_flux` for zero
/// mesh velocity. `normal` must be a unit vector.
///
/// # Examples
///
/// ```rust
/// use sonium_fem::physics::fluid::ale::ale_rusanov_flux;
/// use sonium_fem::physics::fluid::compressible_flow::{
///     ConservativeState, IdealGas, rusanov_flux,
/// };
///
/// let gas = IdealGas::AIR;
/// let left = ConservativeState::from_primitive(1.2, [30.0, 0.0, 0.0], 101_325.0, gas);
/// let right = ConservativeState::from_primitive(1.0, [0.0, 0.0, 0.0], 90_000.0, gas);
/// let normal = [1.0, 0.0, 0.0];
/// assert_eq!(
///     ale_rusanov_flux(left, right, gas, normal, 0.0),
///     rusanov_flux(left, right, gas, normal)
/// );
/// ```
pub fn ale_rusanov_flux(
    left: ConservativeState,
    right: ConservativeState,
    gas: IdealGas,
    normal: [f64; 3],
    mesh_normal_velocity: f64,
) -> [f64; 5] {
    let left_flux = left.euler_flux(gas, normal);
    let right_flux = right.euler_flux(gas, normal);
    let left_array = left.as_array();
    let right_array = right.as_array();
    let left_relative = dot3(left.velocity(), normal) - mesh_normal_velocity;
    let right_relative = dot3(right.velocity(), normal) - mesh_normal_velocity;
    let left_speed = left_relative.abs() + gas.speed_of_sound(left.pressure(gas), left.density);
    let right_speed = right_relative.abs() + gas.speed_of_sound(right.pressure(gas), right.density);
    let maximum = left_speed.max(right_speed);
    let mut flux = [0.0; 5];
    for equation in 0..5 {
        flux[equation] = 0.5 * (left_flux[equation] + right_flux[equation])
            - 0.5 * mesh_normal_velocity * (left_array[equation] + right_array[equation])
            - 0.5 * maximum * (right_array[equation] - left_array[equation]);
    }
    flux
}

/// Advances cell means over one ALE step.
///
/// Moves from `old` to `new` grid in `dt` with slip walls on all
/// boundaries (wall velocity follows boundary nodes). Returns the new
/// cell means; uniform states are preserved to roundoff.
///
/// # Examples
///
/// ```rust
/// use sonium_fem::physics::fluid::ale::{QuadGrid, ale_step};
/// use sonium_fem::physics::fluid::compressible_flow::{ConservativeState, IdealGas};
///
/// let gas = IdealGas::AIR;
/// let old = QuadGrid::rectangle(2, 2, 0.0, 1.0, 0.0, 1.0).unwrap();
/// let state = vec![
///     ConservativeState::from_primitive(1.2, [0.0; 3], 101_325.0, gas);
///     old.num_cells()
/// ];
/// let next = ale_step(&old, &old, &state, gas, 1.0e-4).unwrap();
/// assert_eq!(next.len(), old.num_cells());
/// ```
///
/// # Errors
///
/// Returns `AleError::InvalidGrid` for topology mismatch or invalid
/// `dt`, `AleError::StateSizeMismatch` for wrong state length,
/// `AleError::NonPositiveVolume` for collapsed cells, and
/// `AleError::NonPhysicalState` for non-positive density/pressure.
pub fn ale_step(
    old: &QuadGrid,
    new: &QuadGrid,
    state: &[ConservativeState],
    gas: IdealGas,
    dt: f64,
) -> Result<Vec<ConservativeState>, AleError> {
    old.check_topology(new)?;
    if !dt.is_finite() || dt <= 0.0 {
        return Err(AleError::InvalidParameter);
    }
    if state.len() != old.num_cells() {
        return Err(AleError::StateSizeMismatch);
    }
    let old_volumes = old.volumes();
    let new_volumes = new.volumes();
    if old_volumes.iter().any(|v| !v.is_finite() || *v <= 0.0)
        || new_volumes.iter().any(|v| !v.is_finite() || *v <= 0.0)
    {
        return Err(AleError::NonPositiveVolume);
    }
    for value in state {
        if !value.density.is_finite()
            || value.density <= 0.0
            || !value.pressure(gas).is_finite()
            || value.pressure(gas) <= 0.0
        {
            return Err(AleError::NonPhysicalState);
        }
    }
    let mut totals: Vec<[f64; 5]> = state
        .iter()
        .zip(old_volumes.iter())
        .map(|(value, volume)| scaled_array(value.as_array(), *volume))
        .collect();

    // Vertical facets: minus (left) cell (i-1, j), plus (right) cell (i, j).
    for j in 0..old.ny() {
        for i in 0..=old.nx() {
            let a_old = old.node(i, j);
            let b_old = old.node(i, j + 1);
            let edge = [b_old[0] - a_old[0], b_old[1] - a_old[1]];
            let length = edge[0].hypot(edge[1]);
            let normal2 = [edge[1] / length, -edge[0] / length];
            let normal = [normal2[0], normal2[1], 0.0];
            let swept = swept_gain(a_old, b_old, new.node(i, j), new.node(i, j + 1));
            let mesh_normal = swept / (length * dt);
            if i == 0 {
                let cell = j * old.nx();
                let outward = [-normal[0], -normal[1], 0.0];
                let flux = wall_flux(state[cell], gas, outward, -mesh_normal);
                add_scaled(&mut totals[cell], flux, -dt * length);
            } else if i == old.nx() {
                let cell = j * old.nx() + old.nx() - 1;
                let flux = wall_flux(state[cell], gas, normal, mesh_normal);
                add_scaled(&mut totals[cell], flux, -dt * length);
            } else {
                let minus = j * old.nx() + i - 1;
                let plus = j * old.nx() + i;
                let flux = ale_rusanov_flux(state[minus], state[plus], gas, normal, mesh_normal);
                add_scaled(&mut totals[minus], flux, -dt * length);
                add_scaled(&mut totals[plus], flux, dt * length);
            }
        }
    }

    // Horizontal facets: minus (below) cell (i, j-1), plus cell (i, j).
    // Directed right-to-left so the minus cell sits on the left.
    for j in 0..=old.ny() {
        for i in 0..old.nx() {
            let a_old = old.node(i + 1, j);
            let b_old = old.node(i, j);
            let edge = [b_old[0] - a_old[0], b_old[1] - a_old[1]];
            let length = edge[0].hypot(edge[1]);
            let normal2 = [edge[1] / length, -edge[0] / length];
            let normal = [normal2[0], normal2[1], 0.0];
            let swept = swept_gain(a_old, b_old, new.node(i + 1, j), new.node(i, j));
            let mesh_normal = swept / (length * dt);
            if j == 0 {
                let cell = i;
                let outward = [-normal[0], -normal[1], 0.0];
                let flux = wall_flux(state[cell], gas, outward, -mesh_normal);
                add_scaled(&mut totals[cell], flux, -dt * length);
            } else if j == old.ny() {
                let cell = (old.ny() - 1) * old.nx() + i;
                let flux = wall_flux(state[cell], gas, normal, mesh_normal);
                add_scaled(&mut totals[cell], flux, -dt * length);
            } else {
                let minus = (j - 1) * old.nx() + i;
                let plus = j * old.nx() + i;
                let flux = ale_rusanov_flux(state[minus], state[plus], gas, normal, mesh_normal);
                add_scaled(&mut totals[minus], flux, -dt * length);
                add_scaled(&mut totals[plus], flux, dt * length);
            }
        }
    }

    Ok(totals
        .into_iter()
        .zip(new_volumes.iter())
        .map(|(total, volume)| ConservativeState::from_array(scaled_array(total, 1.0 / volume)))
        .collect())
}

/// Remaps cell means between two grids conservatively.
///
/// Donor-cell transport through swept volumes: exact for uniform
/// fields, conservative whenever boundary nodes stay fixed (the
/// standard split-ALE case). Moving boundary swept volumes fill with
/// the adjacent cell state.
///
/// # Examples
///
/// ```rust
/// use sonium_fem::physics::fluid::ale::{QuadGrid, remap_donor};
/// use sonium_fem::physics::fluid::compressible_flow::{ConservativeState, IdealGas};
///
/// let gas = IdealGas::AIR;
/// let old = QuadGrid::rectangle(2, 2, 0.0, 1.0, 0.0, 1.0).unwrap();
/// let state = vec![
///     ConservativeState::from_primitive(1.0, [0.0; 3], 1.0e5, gas);
///     old.num_cells()
/// ];
/// let next = remap_donor(&old, &old, &state).unwrap();
/// assert_eq!(next, state);
/// ```
///
/// # Errors
///
/// Returns `AleError::InvalidGrid` for topology mismatch,
/// `AleError::StateSizeMismatch` for wrong state length, and
/// `AleError::NonPositiveVolume` for collapsed cells.
pub fn remap_donor(
    old: &QuadGrid,
    new: &QuadGrid,
    state: &[ConservativeState],
) -> Result<Vec<ConservativeState>, AleError> {
    old.check_topology(new)?;
    if state.len() != old.num_cells() {
        return Err(AleError::StateSizeMismatch);
    }
    let new_volumes = new.volumes();
    if old.volumes().iter().any(|v| !v.is_finite() || *v <= 0.0)
        || new_volumes.iter().any(|v| !v.is_finite() || *v <= 0.0)
    {
        return Err(AleError::NonPositiveVolume);
    }
    let mut totals: Vec<[f64; 5]> = state
        .iter()
        .zip(old.volumes().iter())
        .map(|(value, volume)| scaled_array(value.as_array(), *volume))
        .collect();

    for j in 0..old.ny() {
        for i in 0..=old.nx() {
            let swept = swept_gain(
                old.node(i, j),
                old.node(i, j + 1),
                new.node(i, j),
                new.node(i, j + 1),
            );
            if i == 0 {
                let cell = j * old.nx();
                // Cell sits on the plus side; outward gain is negated.
                add_scaled(&mut totals[cell], state[cell].as_array(), -swept);
            } else if i == old.nx() {
                let cell = j * old.nx() + old.nx() - 1;
                add_scaled(&mut totals[cell], state[cell].as_array(), swept);
            } else {
                let minus = j * old.nx() + i - 1;
                let plus = j * old.nx() + i;
                // Positive swept volume flows from plus into minus.
                let donor = if swept >= 0.0 { plus } else { minus };
                add_scaled(&mut totals[minus], state[donor].as_array(), swept);
                add_scaled(&mut totals[plus], state[donor].as_array(), -swept);
            }
        }
    }
    for j in 0..=old.ny() {
        for i in 0..old.nx() {
            // Directed right-to-left so the minus cell sits on the left.
            let swept = swept_gain(
                old.node(i + 1, j),
                old.node(i, j),
                new.node(i + 1, j),
                new.node(i, j),
            );
            if j == 0 {
                let cell = i;
                add_scaled(&mut totals[cell], state[cell].as_array(), -swept);
            } else if j == old.ny() {
                let cell = (old.ny() - 1) * old.nx() + i;
                add_scaled(&mut totals[cell], state[cell].as_array(), swept);
            } else {
                let minus = (j - 1) * old.nx() + i;
                let plus = j * old.nx() + i;
                let donor = if swept >= 0.0 { plus } else { minus };
                add_scaled(&mut totals[minus], state[donor].as_array(), swept);
                add_scaled(&mut totals[plus], state[donor].as_array(), -swept);
            }
        }
    }

    Ok(totals
        .into_iter()
        .zip(new_volumes.iter())
        .map(|(total, volume)| ConservativeState::from_array(scaled_array(total, 1.0 / volume)))
        .collect())
}

/// Relaxes interior nodes toward neighbor averages.
///
/// Applies `sweeps` Laplacian passes with under-relaxation `omega`
/// in `(0, 1]`; masked nodes stay fixed. Mirrors DONEA-style grid
/// smoothing (`/ALE/GRID/DONEA` Donea coefficient 0.05 in bar2_v41).
///
/// # Examples
///
/// ```rust
/// use sonium_fem::physics::fluid::ale::{QuadGrid, laplacian_smooth};
///
/// let grid = QuadGrid::rectangle(3, 3, 0.0, 1.0, 0.0, 1.0).unwrap();
/// // Uniform grids are already relaxed fixed-point states.
/// let same = laplacian_smooth(&grid, None, 1.0, 4).unwrap();
/// assert_eq!(same, grid);
/// ```
///
/// # Errors
///
/// Returns `AleError::InvalidParameter` for `omega` outside `(0, 1]`
/// and `AleError::InvalidGrid` for a mis-sized fixed mask.
pub fn laplacian_smooth(
    grid: &QuadGrid,
    fixed: Option<&[bool]>,
    omega: f64,
    sweeps: usize,
) -> Result<QuadGrid, AleError> {
    if !omega.is_finite() || omega <= 0.0 || omega > 1.0 {
        return Err(AleError::InvalidParameter);
    }
    if fixed.is_some_and(|mask| mask.len() != grid.num_nodes()) {
        return Err(AleError::InvalidGrid);
    }
    let fixed_at = |i: usize, j: usize| -> bool {
        if i == 0 || j == 0 || i == grid.nx() || j == grid.ny() {
            return true;
        }
        fixed.is_some_and(|mask| mask[j * (grid.nx() + 1) + i])
    };
    let mut relaxed = grid.clone();
    for _ in 0..sweeps {
        let previous = relaxed.clone();
        for j in 1..grid.ny() {
            for i in 1..grid.nx() {
                if fixed_at(i, j) {
                    continue;
                }
                let average = [
                    (previous.node(i - 1, j)[0]
                        + previous.node(i + 1, j)[0]
                        + previous.node(i, j - 1)[0]
                        + previous.node(i, j + 1)[0])
                        / 4.0,
                    (previous.node(i - 1, j)[1]
                        + previous.node(i + 1, j)[1]
                        + previous.node(i, j - 1)[1]
                        + previous.node(i, j + 1)[1])
                        / 4.0,
                ];
                let current = previous.node(i, j);
                relaxed.set_node(
                    i,
                    j,
                    [
                        current[0] + omega * (average[0] - current[0]),
                        current[1] + omega * (average[1] - current[1]),
                    ],
                );
            }
        }
    }
    Ok(relaxed)
}

/// Slip-wall ALE flux along an outward unit normal.
///
/// Mirrors the interior velocity about the wall normal motion, so the
/// mass flux vanishes exactly and stationary walls stay adiabatic.
fn wall_flux(
    interior: ConservativeState,
    gas: IdealGas,
    outward: [f64; 3],
    wall_normal_velocity: f64,
) -> [f64; 5] {
    let velocity = interior.velocity();
    let relative = dot3(velocity, outward) - wall_normal_velocity;
    let mirrored = [
        velocity[0] - 2.0 * relative * outward[0],
        velocity[1] - 2.0 * relative * outward[1],
        velocity[2] - 2.0 * relative * outward[2],
    ];
    let ghost =
        ConservativeState::from_primitive(interior.density, mirrored, interior.pressure(gas), gas);
    ale_rusanov_flux(interior, ghost, gas, outward, wall_normal_velocity)
}

fn scaled_array(values: [f64; 5], scale: f64) -> [f64; 5] {
    [
        values[0] * scale,
        values[1] * scale,
        values[2] * scale,
        values[3] * scale,
        values[4] * scale,
    ]
}

fn add_scaled(total: &mut [f64; 5], flux: [f64; 5], scale: f64) {
    for equation in 0..5 {
        total[equation] += flux[equation] * scale;
    }
}

fn dot3(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Periodic 1D linear advection solver.
///
/// Second-order minmod-limited finite volumes with Heun time steps;
/// the Eulerian-limit check of the ALE transport (`ale_advection`
/// fixture replays exact translation through this path).
#[derive(Debug, Clone, PartialEq)]
pub struct Advection1d {
    cells: usize,
    length: f64,
    velocity: f64,
    values: Vec<f64>,
}

impl Advection1d {
    /// Builds a periodic grid sampling `initial` at centers.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use sonium_fem::physics::fluid::ale::Advection1d;
    ///
    /// let solver = Advection1d::from_function(8, 2.0, 1.0, |x| x).unwrap();
    /// assert_eq!(solver.values().len(), 8);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns `AleError::InvalidParameter` for fewer than three
    /// cells, non-positive length, or non-finite velocity.
    pub fn from_function(
        cells: usize,
        length: f64,
        velocity: f64,
        initial: impl Fn(f64) -> f64,
    ) -> Result<Self, AleError> {
        if cells < 3 || !length.is_finite() || length <= 0.0 || !velocity.is_finite() {
            return Err(AleError::InvalidParameter);
        }
        let width = length / cells as f64;
        let values = (0..cells)
            .map(|cell| initial((cell as f64 + 0.5) * width))
            .collect();
        Ok(Self {
            cells,
            length,
            velocity,
            values,
        })
    }

    /// Returns the current cell means.
    pub fn values(&self) -> &[f64] {
        &self.values
    }

    /// Returns the cell width.
    pub fn cell_width(&self) -> f64 {
        self.length / self.cells as f64
    }

    /// Advances one Heun step at the given CFL number.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use sonium_fem::physics::fluid::ale::Advection1d;
    ///
    /// let mut solver = Advection1d::from_function(16, 1.0, 1.0, |_| 2.0).unwrap();
    /// solver.step(0.5).unwrap();
    /// assert!(solver.values().iter().all(|v| (v - 2.0).abs() < 1.0e-15));
    /// ```
    ///
    /// # Errors
    ///
    /// Returns `AleError::InvalidParameter` for CFL outside `(0, 1)`
    /// and `AleError::NonPhysicalState` for non-finite values.
    pub fn step(&mut self, cfl: f64) -> Result<f64, AleError> {
        if !cfl.is_finite() || cfl <= 0.0 || cfl >= 1.0 {
            return Err(AleError::InvalidParameter);
        }
        if self.values.iter().any(|v| !v.is_finite()) {
            return Err(AleError::NonPhysicalState);
        }
        let dt = cfl * self.cell_width() / self.velocity.abs().max(f64::MIN_POSITIVE);
        self.heun_update(dt)?;
        Ok(dt)
    }

    /// Advances to `time` with CFL-limited Heun steps.
    ///
    /// The last step lands exactly on `time`.
    ///
    /// # Errors
    ///
    /// Returns `AleError::InvalidParameter` for non-positive `time`
    /// or CFL outside `(0, 1)`.
    pub fn run(&mut self, time: f64, cfl: f64) -> Result<(), AleError> {
        if !time.is_finite() || time <= 0.0 {
            return Err(AleError::InvalidParameter);
        }
        if !cfl.is_finite() || cfl <= 0.0 || cfl >= 1.0 {
            return Err(AleError::InvalidParameter);
        }
        let rate = self.velocity.abs().max(f64::MIN_POSITIVE) / self.cell_width();
        let steps = ((time * rate / cfl).ceil() as usize).max(1);
        let dt = time / steps as f64;
        for _ in 0..steps {
            self.heun_update(dt)?;
        }
        Ok(())
    }

    /// Returns the maximum error against `exact` at centers.
    pub fn max_error(&self, exact: impl Fn(f64) -> f64) -> f64 {
        let width = self.cell_width();
        self.values
            .iter()
            .enumerate()
            .map(|(cell, value)| (value - exact((cell as f64 + 0.5) * width)).abs())
            .fold(0.0, f64::max)
    }

    fn heun_update(&mut self, dt: f64) -> Result<(), AleError> {
        if self.values.iter().any(|v| !v.is_finite()) {
            return Err(AleError::NonPhysicalState);
        }
        let predictor = self.euler_update(dt);
        let saved = std::mem::replace(&mut self.values, predictor);
        let corrected = self.euler_update(dt);
        for (value, (old, new)) in self
            .values
            .iter_mut()
            .zip(saved.iter().zip(corrected.iter()))
        {
            *value = 0.5 * (old + new);
        }
        Ok(())
    }

    fn euler_update(&self, dt: f64) -> Vec<f64> {
        let width = self.cell_width();
        let factor = dt * self.velocity / width;
        let slopes: Vec<f64> = (0..self.cells)
            .map(|cell| {
                let backward =
                    self.values[cell] - self.values[(cell + self.cells - 1) % self.cells];
                let forward = self.values[(cell + 1) % self.cells] - self.values[cell];
                minmod(backward, forward) / width
            })
            .collect();
        (0..self.cells)
            .map(|cell| {
                let inflow = self.facet_value((cell + self.cells - 1) % self.cells, &slopes);
                let outflow = self.facet_value(cell, &slopes);
                self.values[cell] - factor * (outflow - inflow)
            })
            .collect()
    }

    /// Reconstructs the upwind right-facet state.
    fn facet_value(&self, cell: usize, slopes: &[f64]) -> f64 {
        let width = self.cell_width();
        if self.velocity >= 0.0 {
            self.values[cell] + slopes[cell] * 0.5 * width
        } else {
            let neighbor = (cell + 1) % self.cells;
            self.values[neighbor] - slopes[neighbor] * 0.5 * width
        }
    }
}

/// Minmod slope limiter.
fn minmod(backward: f64, forward: f64) -> f64 {
    if backward * forward <= 0.0 {
        0.0
    } else if backward.abs() < forward.abs() {
        backward
    } else {
        forward
    }
}

/// 1D elastic bar normal-impact reference.
///
/// Closed-form d'Alembert solution for a bar of length `length`
/// striking a rigid wall at `x = 0` with speed `speed`: wall face
/// fixed, far face free, contact stress `rho c v`, separation at
/// `2L/c` with rigid rebound. Lagrangian-limit check of the ALE
/// transport anchored to the bar2_v41 copper deck.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BarImpact {
    /// Mass density in kg/m^3.
    pub density: f64,
    /// Young's modulus in Pa.
    pub young: f64,
    /// Bar length in m.
    pub length: f64,
    /// Impact speed in m/s.
    pub speed: f64,
}

impl BarImpact {
    /// Builds a bar from positive material parameters.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use sonium_fem::physics::fluid::ale::BarImpact;
    ///
    /// let bar = BarImpact::new(8930.0, 117.0e9, 0.0324, 227.0).unwrap();
    /// assert!(bar.wave_speed() > 3600.0);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns `AleError::InvalidParameter` for non-positive or
    /// non-finite inputs.
    pub fn new(density: f64, young: f64, length: f64, speed: f64) -> Result<Self, AleError> {
        if [density, young, length, speed]
            .iter()
            .any(|v| !v.is_finite() || *v <= 0.0)
        {
            return Err(AleError::InvalidParameter);
        }
        Ok(Self {
            density,
            young,
            length,
            speed,
        })
    }

    /// Builds the bar2_v41 copper-bar analog.
    ///
    /// Density 8930 kg/m^3, modulus 117 GPa, length 0.0324 m, and
    /// 227 m/s come from the `BAR2V41B_0000.rad` cards (`/MAT/LAW3`
    /// RHO_0 8.93 with E 0.117 in cm-g-us units, `/INIVEL` Vz
    /// -0.0227, node z extent 0..3.24).
    ///
    /// # Examples
    ///
    /// ```rust
    /// use sonium_fem::physics::fluid::ale::BarImpact;
    ///
    /// let bar = BarImpact::copper_bar2();
    /// assert!((bar.separation_time() - 2.0 * 0.0324 / bar.wave_speed()).abs() < 1.0e-18);
    /// ```
    pub fn copper_bar2() -> Self {
        Self {
            density: 8930.0,
            young: 117.0e9,
            length: 0.0324,
            speed: 227.0,
        }
    }

    /// Returns the longitudinal wave speed `sqrt(E/rho)`.
    pub fn wave_speed(&self) -> f64 {
        (self.young / self.density).sqrt()
    }

    /// Returns the wall contact stress magnitude `rho c v`.
    pub fn contact_stress(&self) -> f64 {
        self.density * self.wave_speed() * self.speed
    }

    /// Returns the separation time `2L/c`.
    pub fn separation_time(&self) -> f64 {
        2.0 * self.length / self.wave_speed()
    }

    /// Returns the axial velocity at `(x, t)`.
    ///
    /// Positions clamp into `[0, length]`; negative times read as
    /// zero; past separation the bar flies rigidly at `+speed`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use sonium_fem::physics::fluid::ale::BarImpact;
    ///
    /// let bar = BarImpact::copper_bar2();
    /// // Wall face stops at first contact.
    /// assert!(bar.velocity(0.0, 1.0e-6).abs() < 1.0e-9);
    /// ```
    pub fn velocity(&self, x: f64, t: f64) -> f64 {
        let time = t.max(0.0);
        if self.wave_speed() * time >= 2.0 * self.length {
            return self.speed;
        }
        let position = x.clamp(0.0, self.length);
        let distance = self.wave_speed() * time;
        0.5 * (self.extended(position - distance) + self.extended(position + distance))
    }

    /// Returns the axial stress at `(x, t)`.
    ///
    /// Negative values are compressive; same clamping as `velocity`,
    /// and zero past separation.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use sonium_fem::physics::fluid::ale::BarImpact;
    ///
    /// let bar = BarImpact::copper_bar2();
    /// // Wall stress equals the contact value before release.
    /// let stress = bar.stress(0.0, 1.0e-6);
    /// assert!((stress + bar.contact_stress()).abs() < 1.0e-3 * bar.contact_stress());
    /// ```
    pub fn stress(&self, x: f64, t: f64) -> f64 {
        let time = t.max(0.0);
        if self.wave_speed() * time >= 2.0 * self.length {
            return 0.0;
        }
        let position = x.clamp(0.0, self.length);
        let distance = self.wave_speed() * time;
        0.5 * self.density
            * self.wave_speed()
            * (self.extended(position + distance) - self.extended(position - distance))
    }

    /// Odd/even image extension of the initial velocity.
    ///
    /// Odd about the wall (`v = 0`) and even about the free face,
    /// giving a `4L`-periodic profile: `+speed` on `[-2L, 0)` and
    /// `-speed` on `[0, 2L)`.
    fn extended(&self, xi: f64) -> f64 {
        let period = 4.0 * self.length;
        let reduced = (xi + 2.0 * self.length).rem_euclid(period) - 2.0 * self.length;
        if reduced < 0.0 {
            self.speed
        } else {
            -self.speed
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::compressible_flow::rusanov_flux;
    use super::*;
    use std::f64::consts::PI;

    /// Deterministic pseudo-random field in `[-1, 1]`.
    fn hash_noise(i: usize, j: usize, seed: u64) -> f64 {
        let mut state = seed
            .wrapping_add((i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15))
            .wrapping_add((j as u64).wrapping_mul(0xBF58_476D_1CE4_E5B9));
        state ^= state >> 30;
        state = state.wrapping_mul(0xBF58_476D_1CE4_E5B9);
        state ^= state >> 27;
        state = state.wrapping_mul(0x94D0_49BB_1331_11EB);
        state ^= state >> 31;
        (state as f64 / u64::MAX as f64) * 2.0 - 1.0
    }

    fn moved_grid(grid: &QuadGrid, amplitude: f64, seed: u64, freeze_walls: bool) -> QuadGrid {
        let mut moved = grid.clone();
        for j in 0..=grid.ny() {
            for i in 0..=grid.nx() {
                let wall = i == 0 || j == 0 || i == grid.nx() || j == grid.ny();
                if freeze_walls && wall {
                    continue;
                }
                let node = grid.node(i, j);
                moved.set_node(
                    i,
                    j,
                    [
                        node[0] + amplitude * hash_noise(i, j, seed),
                        node[1] + amplitude * hash_noise(i, j, seed + 1),
                    ],
                );
            }
        }
        moved
    }

    fn totals(states: &[ConservativeState], volumes: &[f64]) -> [f64; 5] {
        let mut total = [0.0; 5];
        for (state, volume) in states.iter().zip(volumes.iter()) {
            for (slot, value) in total.iter_mut().zip(state.as_array()) {
                *slot += value * volume;
            }
        }
        total
    }

    #[test]
    fn rectangle_volumes_tile_the_domain() {
        let grid = QuadGrid::rectangle(2, 3, 0.0, 2.0, 1.0, 4.0).unwrap();
        let area: f64 = grid.volumes().iter().sum();
        assert!((area - 6.0).abs() < 1.0e-12);
        let center = grid.cell_centroid(0);
        assert!((center[0] - 0.5).abs() < 1.0e-12);
        assert!((center[1] - 1.5).abs() < 1.0e-12);
    }

    #[test]
    fn rectangle_rejects_degenerate_input() {
        assert_eq!(
            QuadGrid::rectangle(0, 2, 0.0, 1.0, 0.0, 1.0).unwrap_err(),
            AleError::InvalidGrid
        );
        assert_eq!(
            QuadGrid::rectangle(2, 2, 1.0, 0.0, 0.0, 1.0).unwrap_err(),
            AleError::InvalidGrid
        );
    }

    #[test]
    fn swept_volumes_tile_cell_volume_changes() {
        // Independent per-cell GCL check: hand-rolled facet balance on
        // one cell with explicit node coordinates, no shared helpers.
        let old = QuadGrid::rectangle(2, 2, 0.0, 1.0, 0.0, 1.0).unwrap();
        let mut new = old.clone();
        new.set_node(1, 1, [0.55, 0.45]);
        new.set_node(1, 2, [0.5, 1.05]);
        let cell = 0;
        let left = swept_gain(
            old.node(0, 0),
            old.node(0, 1),
            new.node(0, 0),
            new.node(0, 1),
        );
        let right = swept_gain(
            old.node(1, 0),
            old.node(1, 1),
            new.node(1, 0),
            new.node(1, 1),
        );
        // Horizontal facets run right-to-left (minus cell on the left).
        let bottom = swept_gain(
            old.node(1, 0),
            old.node(0, 0),
            new.node(1, 0),
            new.node(0, 0),
        );
        let top = swept_gain(
            old.node(1, 1),
            old.node(0, 1),
            new.node(1, 1),
            new.node(0, 1),
        );
        // Cell (0, 0) sits minus-side of right/top, plus-side of left/bottom.
        let balance = -left + right - bottom + top;
        assert!(
            (balance - (new.cell_volume(cell) - old.cell_volume(cell))).abs() < 1.0e-14,
            "GCL balance mismatch"
        );
    }

    #[test]
    fn ale_flux_matches_eulerian_rusanov_at_rest() {
        let gas = IdealGas::AIR;
        let states = [
            ConservativeState::from_primitive(1.2, [30.0, -5.0, 0.0], 101_325.0, gas),
            ConservativeState::from_primitive(0.8, [120.0, 40.0, 0.0], 60_000.0, gas),
        ];
        for normal in [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]] {
            assert_eq!(
                ale_rusanov_flux(states[0], states[1], gas, normal, 0.0),
                rusanov_flux(states[0], states[1], gas, normal)
            );
        }
    }

    #[test]
    fn ale_flux_is_lagrangian_for_matching_mesh_motion() {
        let gas = IdealGas::AIR;
        let flow = ConservativeState::from_primitive(1.5, [50.0, 10.0, 0.0], 80_000.0, gas);
        let normal = [1.0, 0.0, 0.0];
        let mesh = dot3(flow.velocity(), normal);
        let flux = ale_rusanov_flux(flow, flow, gas, normal, mesh);
        assert!(flux[0].abs() < 1.0e-11);
        assert!((flux[1] - 80_000.0).abs() < 1.0e-9);
        assert!(flux[2].abs() < 1.0e-11);
    }

    #[test]
    fn uniform_flow_survives_interior_mesh_motion() {
        // Fixed walls: moving walls are physical pistons, so the
        // free-stream check perturbs interior nodes only.
        let gas = IdealGas::AIR;
        let old = QuadGrid::rectangle(4, 3, 0.0, 1.0, 0.0, 1.0).unwrap();
        let moved = moved_grid(&old, 0.02, 7, true);
        let reference = ConservativeState::from_primitive(1.2, [0.0; 3], 101_325.0, gas);
        let state = vec![reference; old.num_cells()];
        let next = ale_step(&old, &moved, &state, gas, 1.0e-4).unwrap();
        for value in &next {
            assert!((value.density - 1.2).abs() < 1.0e-12);
            assert!(value.velocity().iter().all(|u| u.abs() < 1.0e-9));
            assert!((value.pressure(gas) - 101_325.0).abs() < 1.0e-7);
        }
    }

    #[test]
    fn mass_survives_moving_slip_walls() {
        let gas = IdealGas::AIR;
        let old = QuadGrid::rectangle(5, 4, 0.0, 2.0, 0.0, 1.0).unwrap();
        let moved = moved_grid(&old, 0.03, 21, false);
        let state: Vec<ConservativeState> = (0..old.num_cells())
            .map(|cell| {
                let center = old.cell_centroid(cell);
                let density = 1.0 + 0.3 * (PI * center[0]).sin() * (PI * center[1]).cos();
                ConservativeState::from_primitive(density, [0.0; 3], 1.0e5, gas)
            })
            .collect();
        let next = ale_step(&old, &moved, &state, gas, 2.0e-4).unwrap();
        let before = totals(&state, &old.volumes());
        let after = totals(&next, &moved.volumes());
        assert!(((after[0] - before[0]) / before[0]).abs() < 1.0e-12);
    }

    #[test]
    fn static_step_conserves_mass_and_energy() {
        let gas = IdealGas::AIR;
        let grid = QuadGrid::rectangle(20, 1, 0.0, 1.0, 0.0, 0.05).unwrap();
        let (left, right) = (
            ConservativeState::from_primitive(1.0, [0.0; 3], 1.0, gas),
            ConservativeState::from_primitive(0.125, [0.0; 3], 0.1, gas),
        );
        let state: Vec<ConservativeState> = (0..grid.num_cells())
            .map(|cell| {
                if grid.cell_centroid(cell)[0] < 0.5 {
                    left
                } else {
                    right
                }
            })
            .collect();
        let next = ale_step(&grid, &grid, &state, gas, 1.0e-4).unwrap();
        let volumes = grid.volumes();
        let before = totals(&state, &volumes);
        let after = totals(&next, &volumes);
        assert!(((after[0] - before[0]) / before[0]).abs() < 1.0e-12);
        assert!(((after[4] - before[4]) / before[4]).abs() < 1.0e-12);
        // The discontinuity must evolve: some cell changes density.
        assert!(
            next.iter()
                .zip(state.iter())
                .any(|(new, old)| (new.density - old.density).abs() > 1.0e-9)
        );
    }

    #[test]
    fn step_rejects_invalid_inputs() {
        let gas = IdealGas::AIR;
        let grid = QuadGrid::rectangle(2, 2, 0.0, 1.0, 0.0, 1.0).unwrap();
        let other = QuadGrid::rectangle(3, 2, 0.0, 1.0, 0.0, 1.0).unwrap();
        let state =
            vec![ConservativeState::from_primitive(1.0, [0.0; 3], 1.0e5, gas); grid.num_cells()];
        assert_eq!(
            ale_step(&grid, &other, &state, gas, 1.0e-4).unwrap_err(),
            AleError::InvalidGrid
        );
        assert_eq!(
            ale_step(&grid, &grid, &state[1..], gas, 1.0e-4).unwrap_err(),
            AleError::StateSizeMismatch
        );
        assert_eq!(
            ale_step(&grid, &grid, &state, gas, 0.0).unwrap_err(),
            AleError::InvalidParameter
        );
        let mut vacuum = state.clone();
        vacuum[0].density = 0.0;
        assert_eq!(
            ale_step(&grid, &grid, &vacuum, gas, 1.0e-4).unwrap_err(),
            AleError::NonPhysicalState
        );
        let mut tangled = grid.clone();
        tangled.set_node(1, 1, [2.0, 2.0]);
        assert!(tangled.cell_volume(0) <= 0.0 || tangled.cell_volume(3) <= 0.0);
        assert_eq!(
            ale_step(&grid, &tangled, &state, gas, 1.0e-4).unwrap_err(),
            AleError::NonPositiveVolume
        );
    }

    #[test]
    fn remap_preserves_uniform_fields() {
        let old = QuadGrid::rectangle(4, 4, 0.0, 1.0, 0.0, 1.0).unwrap();
        let moved = moved_grid(&old, 0.02, 3, false);
        let gas = IdealGas::AIR;
        let reference = ConservativeState::from_primitive(2.0, [10.0, 0.0, 0.0], 2.0e5, gas);
        let state = vec![reference; old.num_cells()];
        let next = remap_donor(&old, &moved, &state).unwrap();
        for value in &next {
            assert!(
                value
                    .as_array()
                    .iter()
                    .zip(reference.as_array())
                    .all(|(got, want)| (got - want).abs() < 1.0e-12 * want.abs().max(1.0))
            );
        }
    }

    #[test]
    fn remap_conserves_with_fixed_boundaries() {
        let old = QuadGrid::rectangle(5, 5, 0.0, 1.0, 0.0, 1.0).unwrap();
        let moved = moved_grid(&old, 0.02, 11, true);
        let gas = IdealGas::AIR;
        let state: Vec<ConservativeState> = (0..old.num_cells())
            .map(|cell| {
                let center = old.cell_centroid(cell);
                let density = 1.0 + 0.5 * (2.0 * PI * center[0]).sin();
                ConservativeState::from_primitive(density, [5.0, 0.0, 0.0], 1.0e5, gas)
            })
            .collect();
        let next = remap_donor(&old, &moved, &state).unwrap();
        let before = totals(&state, &old.volumes());
        let after = totals(&next, &moved.volumes());
        for equation in 0..5 {
            assert!(
                (after[equation] - before[equation]).abs()
                    < 1.0e-12 * before[equation].abs().max(1.0),
                "equation {equation} not conserved"
            );
        }
        let minimum = state
            .iter()
            .map(|s| s.density)
            .fold(f64::INFINITY, f64::min);
        let maximum = state
            .iter()
            .map(|s| s.density)
            .fold(f64::NEG_INFINITY, f64::max);
        assert!(
            next.iter()
                .all(|s| s.density >= minimum - 1.0e-12 && s.density <= maximum + 1.0e-12)
        );
    }

    #[test]
    fn smoothing_holds_boundaries_and_relaxes_interior() {
        let grid = QuadGrid::rectangle(4, 4, 0.0, 1.0, 0.0, 1.0).unwrap();
        let mut perturbed = grid.clone();
        perturbed.set_node(2, 2, [0.6, 0.4]);
        let relaxed = laplacian_smooth(&perturbed, None, 1.0, 20).unwrap();
        for i in 0..=grid.nx() {
            assert_eq!(relaxed.node(i, 0), grid.node(i, 0));
            assert_eq!(relaxed.node(i, grid.ny()), grid.node(i, grid.ny()));
        }
        for j in 0..=grid.ny() {
            assert_eq!(relaxed.node(0, j), grid.node(0, j));
            assert_eq!(relaxed.node(grid.nx(), j), grid.node(grid.nx(), j));
        }
        let before = (perturbed.node(2, 2)[0] - 0.5).hypot(perturbed.node(2, 2)[1] - 0.5);
        let after = (relaxed.node(2, 2)[0] - 0.5).hypot(relaxed.node(2, 2)[1] - 0.5);
        assert!(after < 0.2 * before);
        assert_eq!(
            laplacian_smooth(&grid, None, 0.0, 1).unwrap_err(),
            AleError::InvalidParameter
        );
        assert_eq!(
            laplacian_smooth(&grid, Some(&[false; 3]), 1.0, 1).unwrap_err(),
            AleError::InvalidGrid
        );
    }

    #[test]
    fn advection_returns_after_one_period() {
        // Minmod clips smooth extrema (locally first order), so one
        // period needs 800 cells for the 5e-3 band; the W fixture
        // uses the same resolution with a documented margin.
        for velocity in [1.0, -1.0] {
            let mut solver =
                Advection1d::from_function(800, 2.0, velocity, |x| (PI * x).sin()).unwrap();
            solver.run(2.0, 0.5).unwrap();
            let error = solver.max_error(|x| (PI * x).sin());
            assert!(error < 5.0e-3, "velocity {velocity}: error {error}");
        }
    }

    #[test]
    fn advection_conserves_mean_and_rejects_bad_input() {
        let mut solver =
            Advection1d::from_function(64, 1.0, 2.0, |x| 1.0 + 0.25 * (2.0 * PI * x).cos())
                .unwrap();
        let before: f64 = solver.values().iter().sum::<f64>() / 64.0;
        solver.run(0.5, 0.8).unwrap();
        let after: f64 = solver.values().iter().sum::<f64>() / 64.0;
        assert!((after - before).abs() < 1.0e-12);
        assert_eq!(solver.step(1.5).unwrap_err(), AleError::InvalidParameter);
        assert!(
            Advection1d::from_function(2, 1.0, 1.0, |x| x).unwrap_err()
                == AleError::InvalidParameter
        );
    }

    #[test]
    fn bar_holds_wall_and_free_face() {
        let bar = BarImpact::copper_bar2();
        let transit = bar.length / bar.wave_speed();
        // Wall face stops; wall stress equals contact compression.
        assert!(bar.velocity(0.0, 0.5 * transit).abs() < 1.0e-9);
        assert!((bar.stress(0.0, 0.5 * transit) + bar.contact_stress()).abs() < 1.0);
        // Free face keeps initial speed with zero stress until arrival.
        assert!((bar.velocity(bar.length, 0.5 * transit) + bar.speed).abs() < 1.0e-9);
        assert!(bar.stress(bar.length, 0.5 * transit).abs() < 1.0);
        // Ahead of the front the bar still moves rigidly.
        assert!((bar.velocity(0.9 * bar.length, 0.5 * transit) + bar.speed).abs() < 1.0e-9);
        assert!(bar.stress(0.9 * bar.length, 0.5 * transit).abs() < 1.0);
    }

    #[test]
    fn bar_separates_with_rigid_rebound() {
        let bar = BarImpact::copper_bar2();
        let separation = bar.separation_time();
        let just_before = separation * (1.0 - 1.0e-9);
        // Interior already rebounds; the wall face stops until release.
        for x in [
            0.25 * bar.length,
            0.5 * bar.length,
            0.75 * bar.length,
            bar.length,
        ] {
            assert!((bar.velocity(x, just_before) - bar.speed).abs() < 1.0e-6);
            assert!(bar.stress(x, just_before).abs() < 1.0e3);
        }
        assert!(bar.velocity(0.0, just_before).abs() < 1.0e-9);
        assert!((bar.stress(0.0, just_before) + bar.contact_stress()).abs() < 1.0);
        for x in [0.0, 0.5 * bar.length, bar.length] {
            assert_eq!(bar.velocity(x, 2.0 * separation), bar.speed);
            assert_eq!(bar.stress(x, 2.0 * separation), 0.0);
        }
    }

    #[test]
    fn bar_conserves_mechanical_energy() {
        let bar = BarImpact::copper_bar2();
        let initial = 0.5 * bar.density * bar.length * bar.speed * bar.speed;
        for time in [
            0.5 * bar.length / bar.wave_speed(),
            1.5 * bar.length / bar.wave_speed(),
        ] {
            let points = 4000;
            let width = bar.length / points as f64;
            let mut energy = 0.0;
            for point in 0..points {
                let x = (point as f64 + 0.5) * width;
                let velocity = bar.velocity(x, time);
                let stress = bar.stress(x, time);
                energy += (0.5 * bar.density * velocity * velocity
                    + stress * stress / (2.0 * bar.young))
                    * width;
            }
            assert!(
                ((energy - initial) / initial).abs() < 5.0e-3,
                "time {time}: energy {energy}, want {initial}"
            );
        }
    }

    #[test]
    fn bar_copper_numbers_match_deck_scales() {
        let bar = BarImpact::copper_bar2();
        assert!((bar.wave_speed() - 3619.0).abs() < 1.0);
        assert!((bar.contact_stress() - 7.34e9).abs() < 0.02e9);
        assert!((bar.separation_time() - 17.9e-6).abs() < 0.1e-6);
    }
}
