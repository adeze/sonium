//! Broad-phase contact search: sweep and bucket candidates.
//!
//! Per-cycle candidate filter for the explicit contact pass, complementing
//! the bounding volume hierarchy in [`super::detection`]. Both produce
//! `(slave node, master triangle)` pairs for the narrow-phase projection in
//! [`super::gap`]; they differ in cost model and calling context:
//!
//! - Use this module per explicit cycle over raw node/facet slices. It needs
//!   no persistent surface object, allocates only the pair list (plus scratch
//!   proportional to the facet count), and returns deterministically sorted
//!   pairs so the penalty scatter is reproducible.
//! - Use the BVH when a persistent [`ContactSurface`](super::detection::ContactSurface)
//!   already exists, for self-contact (the BVH carries the topological
//!   adjacency filter this module lacks), or when one tree is reused across
//!   several narrow-phase passes.
//!
//! Theory reference: Altair Radioss 2022 Theory Manual, "Algorithms for
//! Impact Candidates" (pp. 241-246): bucket sort is linear for regular
//! meshes, recursive binary space partition ("quick sort") is `N log N` and
//! robust to irregular meshes. The sweep below is the one-axis analogue of
//! that quick-sort family; the bucket grid follows the manual's box sort
//! (facets may overlap several boxes).
//!
//! # Examples
//!
//! ```rust
//! use sonium_fem::physics::solid::contact::broad_phase::{BroadPhaseMethod, contact_candidates};
//!
//! let slaves = vec![[0.5, 0.5, 0.01]];
//! let facets = vec![[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]];
//! let pairs = contact_candidates(&slaves, &facets, 0.05, BroadPhaseMethod::Auto);
//! assert_eq!(pairs, vec![(0, 0)]);
//! ```

// Rust guideline compliant 2026-02-21

use std::collections::HashMap;

/// Search strategy for [`contact_candidates`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BroadPhaseMethod {
    /// Sort-and-sweep along the longest global axis.
    Sweep,
    /// Uniform-grid bucket sort over inflated facet boxes.
    Bucket,
    /// Bucket above the crossover, sweep below it.
    #[default]
    Auto,
}

/// Problem size above which [`BroadPhaseMethod::Auto`] picks buckets.
///
/// Sweep predecessors scan from the sorted head per slave, so sweep wins
/// while the pair matrix fits in cache; buckets pay a hash/grid setup that
/// amortizes on larger problems. The value only affects speed: every method
/// returns the same pair set.
const AUTO_CROSSOVER_PAIRS: usize = 4096;

/// Candidate pairs within `search_radius` of master facets.
///
/// A pair is emitted when the slave point falls inside the facet's
/// axis-aligned bounding box inflated by `search_radius` on every axis (a
/// conservative per-axis distance test; the narrow phase rechecks with the
/// exact projection). Output is sorted by `(slave, facet)`.
///
/// # Panics
///
/// Panics when `search_radius` is negative or not finite.
///
/// # Examples
///
/// ```rust
/// use sonium_fem::physics::solid::contact::broad_phase::{BroadPhaseMethod, contact_candidates};
///
/// let slaves = vec![[0.5, 0.5, 1.0]];
/// let facets = vec![[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]];
/// assert!(contact_candidates(&slaves, &facets, 0.05, BroadPhaseMethod::Auto).is_empty());
/// ```
pub fn contact_candidates(
    slaves: &[[f64; 3]],
    facets: &[[[f64; 3]; 3]],
    search_radius: f64,
    method: BroadPhaseMethod,
) -> Vec<(usize, usize)> {
    assert!(
        search_radius >= 0.0 && search_radius.is_finite(),
        "search radius must be finite and non-negative"
    );
    let resolved = match method {
        BroadPhaseMethod::Auto
            if slaves.len().saturating_mul(facets.len()) > AUTO_CROSSOVER_PAIRS =>
        {
            BroadPhaseMethod::Bucket
        }
        BroadPhaseMethod::Auto => BroadPhaseMethod::Sweep,
        other => other,
    };
    match resolved {
        BroadPhaseMethod::Sweep | BroadPhaseMethod::Auto => {
            sweep_candidates(slaves, facets, search_radius)
        }
        BroadPhaseMethod::Bucket => bucket_candidates(slaves, facets, search_radius),
    }
}

/// Sweep candidate search along the longest axis.
///
/// Sorts facet boxes by their minimum on the longest global axis, then scans
/// each slave forward from the sorted head, breaking once box minima pass
/// out of reach. Same pair contract as [`contact_candidates`].
///
/// # Panics
///
/// Panics when `search_radius` is negative or not finite.
pub fn sweep_candidates(
    slaves: &[[f64; 3]],
    facets: &[[[f64; 3]; 3]],
    search_radius: f64,
) -> Vec<(usize, usize)> {
    assert!(
        search_radius >= 0.0 && search_radius.is_finite(),
        "search radius must be finite and non-negative"
    );
    if slaves.is_empty() || facets.is_empty() {
        return Vec::new();
    }
    let boxes: Vec<FacetBox> = facets.iter().map(FacetBox::of_triangle).collect();
    let axis = longest_axis(slaves, &boxes);
    let mut order: Vec<usize> = (0..facets.len()).collect();
    order.sort_by(|&a, &b| {
        boxes[a].min[axis]
            .partial_cmp(&boxes[b].min[axis])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut pairs = Vec::new();
    for (slave_idx, slave) in slaves.iter().enumerate() {
        let reach = slave[axis] + search_radius;
        for &facet_idx in &order {
            let facet_box = &boxes[facet_idx];
            if facet_box.min[axis] > reach {
                break;
            }
            if facet_box.max[axis] < slave[axis] - search_radius {
                continue;
            }
            if facet_box.contains_point(slave, search_radius) {
                pairs.push((slave_idx, facet_idx));
            }
        }
    }
    pairs.sort_unstable();
    pairs
}

/// Bucket candidate search over a uniform grid.
///
/// Inserts each facet into every grid cell its inflated box overlaps, then
/// tests each slave against its own cell only. Same pair contract as
/// [`contact_candidates`]. Cell width targets one facet per cell on average
/// and never drops below twice the search radius, so neighbors always share
/// the slave's cell.
///
/// # Panics
///
/// Panics when `search_radius` is negative or not finite.
pub fn bucket_candidates(
    slaves: &[[f64; 3]],
    facets: &[[[f64; 3]; 3]],
    search_radius: f64,
) -> Vec<(usize, usize)> {
    assert!(
        search_radius >= 0.0 && search_radius.is_finite(),
        "search radius must be finite and non-negative"
    );
    if slaves.is_empty() || facets.is_empty() {
        return Vec::new();
    }
    let boxes: Vec<FacetBox> = facets.iter().map(FacetBox::of_triangle).collect();
    let (grid_min, cell) = grid_layout(slaves, &boxes, search_radius);
    let mut grid: HashMap<[i64; 3], Vec<usize>> = HashMap::new();
    for (facet_idx, facet_box) in boxes.iter().enumerate() {
        let lo = cell_index(
            [
                facet_box.min[0] - search_radius,
                facet_box.min[1] - search_radius,
                facet_box.min[2] - search_radius,
            ],
            &grid_min,
            cell,
        );
        let hi = cell_index(
            [
                facet_box.max[0] + search_radius,
                facet_box.max[1] + search_radius,
                facet_box.max[2] + search_radius,
            ],
            &grid_min,
            cell,
        );
        for ix in lo[0]..=hi[0] {
            for iy in lo[1]..=hi[1] {
                for iz in lo[2]..=hi[2] {
                    grid.entry([ix, iy, iz]).or_default().push(facet_idx);
                }
            }
        }
    }
    let mut pairs = Vec::new();
    for (slave_idx, slave) in slaves.iter().enumerate() {
        let cell_id = cell_index(*slave, &grid_min, cell);
        if let Some(bucket) = grid.get(&cell_id) {
            for &facet_idx in bucket {
                if boxes[facet_idx].contains_point(slave, search_radius) {
                    pairs.push((slave_idx, facet_idx));
                }
            }
        }
    }
    pairs.sort_unstable();
    pairs.dedup();
    pairs
}

/// Axis-aligned box of one triangular facet.
#[derive(Debug, Clone, Copy)]
struct FacetBox {
    min: [f64; 3],
    max: [f64; 3],
}

impl FacetBox {
    /// Bounding box of a triangle's vertices.
    fn of_triangle(triangle: &[[f64; 3]; 3]) -> Self {
        let mut min = [f64::INFINITY; 3];
        let mut max = [f64::NEG_INFINITY; 3];
        for vertex in triangle {
            for axis in 0..3 {
                min[axis] = min[axis].min(vertex[axis]);
                max[axis] = max[axis].max(vertex[axis]);
            }
        }
        Self { min, max }
    }

    /// Point-in-box test with uniform inflation.
    fn contains_point(&self, point: &[f64; 3], inflate: f64) -> bool {
        (0..3).all(|axis| {
            point[axis] >= self.min[axis] - inflate && point[axis] <= self.max[axis] + inflate
        })
    }
}

/// Longest global axis over slaves and facet boxes.
fn longest_axis(slaves: &[[f64; 3]], boxes: &[FacetBox]) -> usize {
    let mut min = [f64::INFINITY; 3];
    let mut max = [f64::NEG_INFINITY; 3];
    for slave in slaves {
        for (axis, value) in slave.iter().enumerate() {
            min[axis] = min[axis].min(*value);
            max[axis] = max[axis].max(*value);
        }
    }
    for facet_box in boxes {
        for (axis, box_min) in facet_box.min.iter().enumerate() {
            min[axis] = min[axis].min(*box_min);
            max[axis] = max[axis].max(facet_box.max[axis]);
        }
    }
    let extents = [max[0] - min[0], max[1] - min[1], max[2] - min[2]];
    if extents[0] >= extents[1] && extents[0] >= extents[2] {
        0
    } else if extents[1] >= extents[2] {
        1
    } else {
        2
    }
}

/// Grid origin and cell width for the bucket search.
fn grid_layout(slaves: &[[f64; 3]], boxes: &[FacetBox], radius: f64) -> ([f64; 3], f64) {
    let mut min = [f64::INFINITY; 3];
    let mut extent: f64 = 0.0;
    for slave in slaves {
        for (axis, value) in slave.iter().enumerate() {
            min[axis] = min[axis].min(*value);
            extent = extent.max(*value - min[axis]);
        }
    }
    for facet_box in boxes {
        for (axis, box_min) in facet_box.min.iter().enumerate() {
            min[axis] = min[axis].min(*box_min);
            extent = extent.max(facet_box.max[axis] - min[axis]);
        }
    }
    // One facet per cell on average along each axis; the radius floor keeps
    // every in-reach facet inside the slave's own cell.
    let per_axis = (boxes.len().max(1) as f64).cbrt().ceil().max(1.0);
    let cell = (extent / per_axis).max(2.0 * radius).max(1.0e-12);
    (min, cell)
}

/// Grid cell containing a point, relative to the grid origin.
fn cell_index(point: [f64; 3], origin: &[f64; 3], cell: f64) -> [i64; 3] {
    [
        ((point[0] - origin[0]) / cell).floor() as i64,
        ((point[1] - origin[1]) / cell).floor() as i64,
        ((point[2] - origin[2]) / cell).floor() as i64,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Reference predicate shared by every method.
    fn brute_force(
        slaves: &[[f64; 3]],
        facets: &[[[f64; 3]; 3]],
        radius: f64,
    ) -> Vec<(usize, usize)> {
        let boxes: Vec<FacetBox> = facets.iter().map(FacetBox::of_triangle).collect();
        let mut pairs = Vec::new();
        for (slave_idx, slave) in slaves.iter().enumerate() {
            for (facet_idx, facet_box) in boxes.iter().enumerate() {
                if facet_box.contains_point(slave, radius) {
                    pairs.push((slave_idx, facet_idx));
                }
            }
        }
        pairs
    }

    /// Deterministic pseudo-random layout without an RNG dependency.
    fn scattered_layout(count: usize) -> (Vec<[f64; 3]>, Vec<[[f64; 3]; 3]>) {
        let mut slaves = Vec::with_capacity(count);
        let mut facets = Vec::with_capacity(count);
        for k in 0..count {
            let t = k as f64;
            // Irrational slopes decorrelate the axes.
            let base = [
                (t * 0.618_033_988_749_895).rem_euclid(4.0) - 2.0,
                (t * std::f64::consts::SQRT_2).rem_euclid(4.0) - 2.0,
                (t * 0.732_050_807_568_877).rem_euclid(1.0),
            ];
            slaves.push([base[0], base[1], base[2] + 0.05]);
            facets.push([
                [base[0] - 0.3, base[1] - 0.3, base[2]],
                [base[0] + 0.3, base[1] - 0.3, base[2]],
                [base[0], base[1] + 0.3, base[2]],
            ]);
        }
        (slaves, facets)
    }

    #[test]
    fn sweep_matches_brute_force() {
        let (slaves, facets) = scattered_layout(60);
        for radius in [0.0, 0.02, 0.2] {
            assert_eq!(
                sweep_candidates(&slaves, &facets, radius),
                brute_force(&slaves, &facets, radius),
                "radius {radius}"
            );
        }
    }

    #[test]
    fn bucket_matches_brute_force() {
        let (slaves, facets) = scattered_layout(60);
        for radius in [0.0, 0.02, 0.2] {
            assert_eq!(
                bucket_candidates(&slaves, &facets, radius),
                brute_force(&slaves, &facets, radius),
                "radius {radius}"
            );
        }
    }

    #[test]
    fn auto_matches_brute_force_on_both_sides_of_crossover() {
        // Below the crossover (sweep) and above it (bucket).
        for count in [8, 80] {
            let (slaves, facets) = scattered_layout(count);
            assert_eq!(
                contact_candidates(&slaves, &facets, 0.1, BroadPhaseMethod::Auto),
                brute_force(&slaves, &facets, 0.1),
                "count {count}"
            );
        }
    }

    #[test]
    fn output_is_sorted_and_deduplicated() {
        let (slaves, facets) = scattered_layout(40);
        for method in [
            BroadPhaseMethod::Sweep,
            BroadPhaseMethod::Bucket,
            BroadPhaseMethod::Auto,
        ] {
            let pairs = contact_candidates(&slaves, &facets, 0.3, method);
            let mut sorted = pairs.clone();
            sorted.sort_unstable();
            sorted.dedup();
            assert_eq!(pairs, sorted, "method {method:?}");
        }
    }

    #[test]
    fn empty_inputs_yield_no_pairs() {
        let facet = [[[0.0; 3]; 3]];
        assert!(sweep_candidates(&[], &facet, 0.1).is_empty());
        assert!(bucket_candidates(&[[0.0; 3]], &[], 0.1).is_empty());
        assert!(contact_candidates(&[], &[], 0.1, BroadPhaseMethod::Auto).is_empty());
    }

    #[test]
    fn distant_slaves_yield_no_pairs() {
        let slaves = vec![[10.0, 10.0, 10.0]];
        let facets = vec![[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]];
        for method in [
            BroadPhaseMethod::Sweep,
            BroadPhaseMethod::Bucket,
            BroadPhaseMethod::Auto,
        ] {
            assert!(
                contact_candidates(&slaves, &facets, 0.5, method).is_empty(),
                "method {method:?}"
            );
        }
    }
}
