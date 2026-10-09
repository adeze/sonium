//! Density and sensitivity filtering for topology optimization
//!
//! Filters prevent checkerboard patterns and mesh-dependent solutions.
//! - **Density filter**: weighted average of neighboring densities
//! - **Sensitivity filter**: weighted average of neighboring sensitivities
//! - **Heaviside projection**: sharpens the density field toward 0/1

use crate::mesh::Mesh;

/// Density/sensitivity filter using weighted spatial averaging.
///
/// For each element, the filtered value is a weighted average of values
/// within a radius `r_min` of the element centroid.
#[derive(Debug, Clone)]
pub struct DensityFilter {
    /// Filter radius (m).
    pub radius: f64,
    /// Precomputed weights: `weights[i]` = vec of `(neighbor_idx, weight)`.
    weights: Vec<Vec<(usize, f64)>>,
}

/// Typed construction failures for [`DensityFilter::build_from_positions`].
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum DensityFilterError {
    /// No positions were supplied.
    #[error("density filter requires positions")]
    EmptyPositions,
    /// The radius is non-finite or non-positive.
    #[error("density filter radius must be finite and positive")]
    InvalidRadius,
    /// The squared radius overflowed to a non-finite value.
    #[error("density filter radius is too large")]
    RadiusTooLarge,
    /// A position coordinate is non-finite.
    #[error("density filter positions must be finite")]
    NonFinitePosition,
    /// A pairwise distance came out non-finite.
    #[error("density filter position distance is non-finite")]
    NonFiniteDistance,
    /// A row sees no support (not even itself), so normalization is undefined.
    #[error("density filter has an empty support row")]
    EmptySupportRow,
}

impl DensityFilter {
    /// Build a density filter from mesh element centroids.
    pub fn build(mesh: &Mesh, radius: f64) -> Self {
        let n = mesh.elements.len();
        let centroids: Vec<[f64; 3]> = (0..n)
            .map(|i| {
                let elem = &mesh.elements[i];
                let verts = elem.vertices();
                let nv = verts.len() as f64;
                let mut cx = 0.0;
                let mut cy = 0.0;
                let mut cz = 0.0;
                for &v in verts {
                    cx += mesh.nodes[v].x;
                    cy += mesh.nodes[v].y;
                    cz += mesh.nodes[v].z;
                }
                [cx / nv, cy / nv, cz / nv]
            })
            .collect();

        let r2 = radius * radius;
        let mut weights = Vec::with_capacity(n);

        for i in 0..n {
            let mut neighbors = Vec::new();
            for j in 0..n {
                let dx = centroids[i][0] - centroids[j][0];
                let dy = centroids[i][1] - centroids[j][1];
                let dz = centroids[i][2] - centroids[j][2];
                let dist2 = dx * dx + dy * dy + dz * dz;

                if dist2 <= r2 {
                    let dist = dist2.sqrt();
                    let w = (radius - dist).max(0.0);
                    if w > 0.0 {
                        neighbors.push((j, w));
                    }
                }
            }

            // Normalize weights
            let total_w: f64 = neighbors.iter().map(|(_, w)| w).sum();
            if total_w > 0.0 {
                for (_, w) in &mut neighbors {
                    *w /= total_w;
                }
            }

            weights.push(neighbors);
        }

        Self { radius, weights }
    }

    /// Build a density filter over a subset of mesh elements (for example the
    /// design cells of a topology problem). Neighbor indices stored in the
    /// weights refer to positions within `elements`, not mesh element indices,
    /// so the filter maps length-`elements.len()` vectors to vectors of the
    /// same length.
    pub fn build_from_elements(mesh: &Mesh, radius: f64, elements: &[usize]) -> Self {
        let centroids: Vec<[f64; 3]> = elements
            .iter()
            .map(|&i| {
                let verts = mesh.elements[i].vertices();
                let nv = verts.len() as f64;
                let mut cx = 0.0;
                let mut cy = 0.0;
                let mut cz = 0.0;
                for &v in verts {
                    cx += mesh.nodes[v].x;
                    cy += mesh.nodes[v].y;
                    cz += mesh.nodes[v].z;
                }
                [cx / nv, cy / nv, cz / nv]
            })
            .collect();

        let r2 = radius * radius;
        let mut weights = Vec::with_capacity(centroids.len());

        for i in 0..centroids.len() {
            let mut neighbors = Vec::new();
            for (j, centroid) in centroids.iter().enumerate() {
                let dx = centroids[i][0] - centroid[0];
                let dy = centroids[i][1] - centroid[1];
                let dz = centroids[i][2] - centroid[2];
                let dist2 = dx * dx + dy * dy + dz * dz;

                if dist2 <= r2 {
                    let dist = dist2.sqrt();
                    let w = (radius - dist).max(0.0);
                    if w > 0.0 {
                        neighbors.push((j, w));
                    }
                }
            }

            // Normalize weights
            let total_w: f64 = neighbors.iter().map(|(_, w)| w).sum();
            if total_w > 0.0 {
                for (_, w) in &mut neighbors {
                    *w /= total_w;
                }
            }

            weights.push(neighbors);
        }

        Self { radius, weights }
    }

    /// Build a checked density filter over an explicit position cloud (for
    /// example panel or treatment centres that are not mesh elements).
    ///
    /// The kernel is the same normalized hat `(radius - dist)` used by
    /// [`Self::build`], so a point cloud sampled at mesh centroids filters
    /// identically. Unlike the infallible mesh constructors, every
    /// degeneracy fails closed: empty clouds, non-finite geometry, an
    /// overflowing radius, and rows with no support (whose normalization
    /// would divide by zero) are all errors.
    pub fn build_from_positions(
        positions: &[[f64; 3]],
        radius: f64,
    ) -> Result<Self, DensityFilterError> {
        if positions.is_empty() {
            return Err(DensityFilterError::EmptyPositions);
        }
        if !radius.is_finite() || radius <= 0.0 {
            return Err(DensityFilterError::InvalidRadius);
        }
        if positions
            .iter()
            .flatten()
            .any(|coordinate| !coordinate.is_finite())
        {
            return Err(DensityFilterError::NonFinitePosition);
        }
        let radius_squared = radius * radius;
        if !radius_squared.is_finite() {
            return Err(DensityFilterError::RadiusTooLarge);
        }

        let mut weights = Vec::with_capacity(positions.len());
        for centre in positions {
            let mut neighbors = Vec::new();
            for (source, position) in positions.iter().enumerate() {
                let distance_squared = centre
                    .iter()
                    .zip(position)
                    .map(|(left, right)| (left - right).powi(2))
                    .sum::<f64>();
                if !distance_squared.is_finite() {
                    return Err(DensityFilterError::NonFiniteDistance);
                }
                if distance_squared <= radius_squared {
                    let weight = (radius - distance_squared.sqrt()).max(0.0);
                    if weight > 0.0 {
                        neighbors.push((source, weight));
                    }
                }
            }
            let row_sum: f64 = neighbors.iter().map(|(_, weight)| weight).sum();
            if !row_sum.is_finite() || row_sum <= 0.0 {
                return Err(DensityFilterError::EmptySupportRow);
            }
            for (_, weight) in &mut neighbors {
                *weight /= row_sum;
            }
            weights.push(neighbors);
        }

        Ok(Self { radius, weights })
    }

    /// Number of filtered positions (rows and columns of the weight map).
    pub fn len(&self) -> usize {
        self.weights.len()
    }

    /// Whether the filter holds no positions.
    pub fn is_empty(&self) -> bool {
        self.weights.is_empty()
    }

    /// Sparse normalized weights of one row: `(source_index, weight)` pairs.
    pub fn row(&self, index: usize) -> &[(usize, f64)] {
        &self.weights[index]
    }

    /// Apply density filter: `rho_filtered[i] = sum_j w_ij * rho[j]`.
    pub fn filter_densities(&self, densities: &[f64]) -> Vec<f64> {
        self.weights
            .iter()
            .map(|neighbors| neighbors.iter().map(|&(j, w)| w * densities[j]).sum())
            .collect()
    }

    /// Chain-rule (transpose) application of the normalized density filter.
    ///
    /// Because [`filter_densities`](Self::filter_densities) computes
    /// `rho_filtered[i] = sum_j w_ij * rho[j]` with rows normalized
    /// (`sum_j w_ij = 1`), the exact adjoint chain rule for objective
    /// sensitivities is the transpose map
    /// `df_design[j] = sum_i w_ij * df_filtered[i]`.
    ///
    /// This differs from [`filter_sensitivities`](Self::filter_sensitivities),
    /// which implements the cone-filter `(sum w*rho*s)/rho` sensitivity
    /// *smoothing* heuristic — a different operation that is not the chain
    /// rule of the normalized density filter.
    pub fn filter_sensitivities_transpose(&self, sensitivities: &[f64]) -> Vec<f64> {
        let mut result = vec![0.0; self.weights.len()];
        for (i, neighbors) in self.weights.iter().enumerate() {
            for &(j, w) in neighbors {
                result[j] += w * sensitivities[i];
            }
        }
        result
    }

    /// Apply sensitivity filter: modifies sensitivities to account for density filtering.
    ///
    /// `df_filtered[i] = (1 / (rho[i] * sum_j w_ij)) * sum_j w_ij * rho[j] * df[j]`
    pub fn filter_sensitivities(&self, densities: &[f64], sensitivities: &[f64]) -> Vec<f64> {
        self.weights
            .iter()
            .enumerate()
            .map(|(i, neighbors)| {
                let rho_i = densities[i].max(1e-9);
                let weighted_sum: f64 = neighbors
                    .iter()
                    .map(|&(j, w)| w * densities[j] * sensitivities[j])
                    .sum();
                weighted_sum / rho_i
            })
            .collect()
    }
}

/// Heaviside projection to push densities toward 0 or 1.
///
/// `rho_proj = (tanh(beta * eta) + tanh(beta * (rho - eta))) /
///             (tanh(beta * eta) + tanh(beta * (1 - eta)))`
///
/// where `beta` controls sharpness (higher = sharper) and `eta` is the threshold (typically 0.5).
#[derive(Debug, Clone, Copy)]
pub struct HeavisideProjection {
    /// Sharpness parameter (start small ~1, increase to ~64 during optimization).
    pub beta: f64,
    /// Threshold (typically 0.5).
    pub eta: f64,
}

impl HeavisideProjection {
    pub fn new(beta: f64, eta: f64) -> Self {
        Self { beta, eta }
    }

    /// Project a density value.
    pub fn project(&self, rho: f64) -> f64 {
        let t1 = (self.beta * self.eta).tanh();
        let t2 = (self.beta * (rho - self.eta)).tanh();
        let t3 = (self.beta * (1.0 - self.eta)).tanh();
        (t1 + t2) / (t1 + t3)
    }

    /// Derivative of projection.
    pub fn project_derivative(&self, rho: f64) -> f64 {
        let t1 = (self.beta * self.eta).tanh();
        let t3 = (self.beta * (1.0 - self.eta)).tanh();
        let sech2 = 1.0 - (self.beta * (rho - self.eta)).tanh().powi(2);
        self.beta * sech2 / (t1 + t3)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_heaviside_limits() {
        let h = HeavisideProjection::new(50.0, 0.5);
        // At rho=0: should be near 0
        assert!(h.project(0.0) < 0.01);
        // At rho=1: should be near 1
        assert!(h.project(1.0) > 0.99);
        // At rho=0.5: should be 0.5
        assert!((h.project(0.5) - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_heaviside_derivative() {
        let h = HeavisideProjection::new(10.0, 0.5);
        let rho = 0.4;
        let eps = 1e-7;
        let numerical = (h.project(rho + eps) - h.project(rho - eps)) / (2.0 * eps);
        let analytical = h.project_derivative(rho);
        assert!(
            (analytical - numerical).abs() < 1e-4,
            "Heaviside derivative: {analytical} vs {numerical}"
        );
    }

    #[test]
    fn test_position_cloud_filter_matches_hat_kernel_math() {
        // Two coincident points: each row sees both at distance 0.
        let filter = DensityFilter::build_from_positions(&[[0.0, 0.0, 0.0]; 2], 1.0).unwrap();
        assert_eq!(filter.filter_densities(&[0.2, 0.8]), vec![0.5, 0.5]);
        // Forward/transpose dot-product identity: <F x, y> == <x, Fᵀ y>.
        let filtered = filter.filter_densities(&[0.2, 0.8]);
        let lhs: f64 = filtered.iter().zip([1.0, 3.0]).map(|(a, b)| a * b).sum();
        let rhs: f64 = [0.2, 0.8]
            .iter()
            .zip(filter.filter_sensitivities_transpose(&[1.0, 3.0]))
            .map(|(a, b)| a * b)
            .sum();
        assert!(
            (lhs - rhs).abs() < 1e-12,
            "adjoint identity: {lhs} vs {rhs}"
        );
    }

    #[test]
    fn test_position_cloud_filter_fails_closed() {
        assert!(matches!(
            DensityFilter::build_from_positions(&[], 1.0),
            Err(DensityFilterError::EmptyPositions)
        ));
        assert!(matches!(
            DensityFilter::build_from_positions(&[[f64::NAN, 0.0, 0.0]], 1.0),
            Err(DensityFilterError::NonFinitePosition)
        ));
        assert!(matches!(
            DensityFilter::build_from_positions(&[[0.0, 0.0, 0.0]], 0.0),
            Err(DensityFilterError::InvalidRadius)
        ));
        assert!(matches!(
            DensityFilter::build_from_positions(&[[0.0, 0.0, 0.0]], f64::INFINITY),
            Err(DensityFilterError::InvalidRadius)
        ));
    }

    #[test]
    fn test_density_filter_uniform() {
        let mesh = crate::mesh::unit_square_triangles(3);
        let filter = DensityFilter::build(&mesh, 0.5);

        // Uniform density should remain uniform after filtering
        let n = mesh.elements.len();
        let rho = vec![0.5; n];
        let filtered = filter.filter_densities(&rho);

        for (i, &r) in filtered.iter().enumerate() {
            assert!(
                (r - 0.5).abs() < 1e-10,
                "Filtered density {i} = {r}, expected 0.5"
            );
        }
    }
}
