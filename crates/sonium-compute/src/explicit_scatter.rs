//! Scatter/gather and lumped-mass assembly for explicit dynamics.
//!
//! CPU kernels behind the Phase 1 lane 1A contract: node-major gather,
//! serial and race-free parallel scatter-add, and lumped-mass assembly
//! over slices shaped like [`crate::explicit_state::ExplicitState`].
//! Everything here is plain `f64` slice code plus `rayon`; there are no
//! sparse-matrix, FEM, or solver types, so the explicit path stays
//! sparse-matrix-free by construction.
//!
//! # Layout
//!
//! Nodal fields are node-major with 0-based node ids: a 3-vector field
//! over `N` nodes is `[x0, y0, z0, x1, ...]` with length `3N`, and a
//! scalar field such as masses has length `N`. All values are SI units
//! (`f64` only, matching the `dp` engine reference).
//!
//! Gather and scatter infer the component width from their arguments:
//! with `M` index entries, the packed buffer (`out` or `values`) must
//! hold exactly `M * width` values, and the nodal buffer must hold an
//! exact multiple of `width`. This one shape covers scalar gathers
//! (`width == 1`, e.g. per-node energy) and vector gathers
//! (`width == 3`, e.g. positions or forces).
//!
//! # Parallelism
//!
//! [`scatter_add_nodal_parallel`] partitions the node-id space into
//! disjoint contiguous ranges (one per `rayon` worker chunk) and lets
//! each worker scan the contribution list in order, accumulating only
//! the entries that land in its range. Writes never overlap, so the
//! reduction is race-free without atomics or locks; each node sees its
//! contributions in the same order as the serial kernel, so the two
//! agree bit-exactly. The trade is read amplification: every worker
//! streams the full index list, so the parallel kernel suits large
//! scatters where the accumulation work dominates the extra reads.
//!
//! # Examples
//!
//! Scatter element forces into a state and assemble its lumped masses:
//!
//! ```rust
//! use sonium_compute::explicit_scatter::{
//!     ElementTopology, assemble_lumped_mass, scatter_add_nodal,
//! };
//! use sonium_compute::explicit_state::ExplicitState;
//!
//! let mut state = ExplicitState::new(2);
//! state.zero_forces();
//! scatter_add_nodal(
//!     &mut state.forces,
//!     &[0, 1],
//!     &[1.0, 0.0, 0.0, 0.0, 2.0, 0.0],
//! );
//! let topology = ElementTopology::new(&[0, 1], &[0, 2]);
//! assemble_lumped_mass(&mut state.masses, topology, &[2.0], &[3.0]);
//! assert!(state.is_consistent());
//! assert_eq!(state.masses, vec![3.0, 3.0]);
//! ```

// Rust guideline compliant 2026-02-21

use rayon::prelude::*;

/// Minimal element topology: CSR-style connectivity.
///
/// Holds the concatenated 0-based node ids of every element plus the
/// per-element offsets into that list. This is the least structure the
/// lumped-mass kernel needs: element `e` owns the nodes in
/// `connectivity[offsets[e]..offsets[e + 1]]`. Mixed element types are
/// fine because each element carries its own node count.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ElementTopology<'a> {
    /// Concatenated 0-based node ids of all elements.
    pub connectivity: &'a [usize],
    /// Start offset of each element into `connectivity`, plus a final
    /// end offset, so `offsets` has length `E + 1` for `E` elements.
    pub offsets: &'a [usize],
}

impl<'a> ElementTopology<'a> {
    /// Validate connectivity and offsets into a topology.
    ///
    /// Offsets must be non-empty, start at 0, and strictly increase
    /// (so every element owns at least one node), with the final offset
    /// inside `connectivity`. Node ids themselves are bounds-checked by
    /// the consuming kernel against its nodal buffers.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use sonium_compute::explicit_scatter::ElementTopology;
    ///
    /// let topology = ElementTopology::new(&[0, 1, 2, 3], &[0, 2, 4]);
    /// assert_eq!(topology.element_count(), 2);
    /// assert_eq!(topology.nodes_of(1), &[2, 3]);
    /// ```
    ///
    /// # Panics
    ///
    /// Panics when `offsets` is empty, does not start at 0, is not
    /// strictly increasing, or runs past the end of `connectivity`.
    #[must_use]
    pub fn new(connectivity: &'a [usize], offsets: &'a [usize]) -> Self {
        assert!(
            !offsets.is_empty(),
            "offsets must hold E + 1 entries, got an empty slice"
        );
        assert!(
            offsets[0] == 0,
            "offsets must start at 0, got {}",
            offsets[0]
        );
        for (element, window) in offsets.windows(2).enumerate() {
            let (start, end) = (window[0], window[1]);
            assert!(
                end > start,
                "element {element} is empty (offsets must strictly increase)"
            );
            assert!(
                end <= connectivity.len(),
                "element {element} end offset {end} exceeds connectivity length {}",
                connectivity.len()
            );
        }
        Self {
            connectivity,
            offsets,
        }
    }

    /// Count elements (`offsets.len() - 1`).
    ///
    /// # Examples
    ///
    /// ```rust
    /// use sonium_compute::explicit_scatter::ElementTopology;
    ///
    /// let topology = ElementTopology::new(&[0, 1, 2], &[0, 3]);
    /// assert_eq!(topology.element_count(), 1);
    /// ```
    ///
    /// # Panics
    ///
    /// Panics when `offsets` is empty, which cannot happen for values
    /// built with [`ElementTopology::new`].
    #[must_use]
    pub fn element_count(&self) -> usize {
        assert!(
            !self.offsets.is_empty(),
            "offsets must hold E + 1 entries, got an empty slice"
        );
        self.offsets.len() - 1
    }

    /// Borrow the node ids of one element.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use sonium_compute::explicit_scatter::ElementTopology;
    ///
    /// let topology = ElementTopology::new(&[0, 1, 2, 3], &[0, 2, 4]);
    /// assert_eq!(topology.nodes_of(0), &[0, 1]);
    /// ```
    ///
    /// # Panics
    ///
    /// Panics when `element` is not a valid element id.
    #[must_use]
    pub fn nodes_of(&self, element: usize) -> &'a [usize] {
        &self.connectivity[self.offsets[element]..self.offsets[element + 1]]
    }
}

/// Validate packed node-major layouts for gather and scatter.
///
/// Returns the inferred component width and node count, or `None` when
/// the index list is empty (a no-op that requires an empty packed
/// buffer). Otherwise the width is `packed_len / indices_len`, the
/// packed buffer must be an exact positive multiple of the index count,
/// and the nodal buffer must be an exact multiple of the width.
///
/// # Panics
///
/// Panics when the packed buffer length is not an exact positive
/// multiple of the index count, or when the nodal buffer length is not
/// an exact multiple of the inferred width.
fn packed_layout(
    nodal_len: usize,
    indices_len: usize,
    packed_len: usize,
) -> Option<(usize, usize)> {
    if indices_len == 0 {
        assert!(
            packed_len == 0,
            "empty index list requires an empty packed buffer, got length {packed_len}"
        );
        return None;
    }
    assert!(
        packed_len >= indices_len && packed_len.is_multiple_of(indices_len),
        "packed length {packed_len} is not an exact positive multiple of index count {indices_len}"
    );
    let width = packed_len / indices_len;
    assert!(
        nodal_len.is_multiple_of(width),
        "nodal length {nodal_len} is not a multiple of inferred width {width}"
    );
    Some((width, nodal_len / width))
}

/// Gather nodal rows into a packed buffer.
///
/// Copies row `indices[m]` of `field` into slot `m` of `out` for every
/// entry, so element kernels can work on dense per-element data. The
/// component width is inferred (see the module docs): gathering `M`
/// 3-vectors needs `out.len() == 3 * M` and a node-major `field`.
///
/// # Examples
///
/// ```rust
/// use sonium_compute::explicit_scatter::gather_nodal;
///
/// let field = vec![10.0, 11.0, 12.0, 20.0, 21.0, 22.0];
/// let mut out = vec![0.0; 6];
/// gather_nodal(&field, &[1, 0], &mut out);
/// assert_eq!(out, vec![20.0, 21.0, 22.0, 10.0, 11.0, 12.0]);
/// ```
///
/// # Panics
///
/// Panics on layout mismatch (packed length not an exact multiple of
/// the index count, or nodal length not a multiple of the width) and
/// when a node id falls outside the nodal buffer.
pub fn gather_nodal(field: &[f64], indices: &[usize], out: &mut [f64]) {
    let Some((width, _nodes)) = packed_layout(field.len(), indices.len(), out.len()) else {
        return;
    };
    for (slot, &node) in indices.iter().enumerate() {
        let src = node * width;
        let dst = slot * width;
        out[dst..dst + width].copy_from_slice(&field[src..src + width]);
    }
}

/// Scatter-add packed rows into a nodal buffer.
///
/// Adds slot `m` of `values` into row `indices[m]` of `target` for
/// every entry in order, so shared nodes accumulate every incident
/// element contribution. This is the serial reference: the parallel
/// kernel below reproduces it bit-exactly.
///
/// # Examples
///
/// ```rust
/// use sonium_compute::explicit_scatter::scatter_add_nodal;
///
/// let mut target = vec![0.0; 6];
/// scatter_add_nodal(&mut target, &[0, 1, 0], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
/// assert_eq!(target, vec![6.0, 8.0, 3.0, 4.0, 0.0, 0.0]);
/// ```
///
/// # Panics
///
/// Panics on layout mismatch (packed length not an exact multiple of
/// the index count, or nodal length not a multiple of the width) and
/// when a node id falls outside the nodal buffer.
pub fn scatter_add_nodal(target: &mut [f64], indices: &[usize], values: &[f64]) {
    let Some((width, _nodes)) = packed_layout(target.len(), indices.len(), values.len()) else {
        return;
    };
    for (slot, &node) in indices.iter().enumerate() {
        let dst = node * width;
        let src = slot * width;
        let dst_row = &mut target[dst..dst + width];
        let src_row = &values[src..src + width];
        for (slot, &value) in dst_row.iter_mut().zip(src_row.iter()) {
            *slot += value;
        }
    }
}

/// Scatter-add packed rows in parallel without races.
///
/// Splits the node-id space into disjoint contiguous ranges, one per
/// `rayon` worker chunk, and each worker scans the contribution list in
/// order, accumulating only the entries that land in its own range.
/// Disjoint writes make the reduction race-free without atomics, and
/// because every node sees its contributions in serial order, the
/// result agrees with [`scatter_add_nodal`] bit-exactly (proven by the
/// in-module tests, including duplicate-heavy sharing patterns).
///
/// The strategy streams the full index list on every worker, so small
/// scatters may run slower than the serial kernel; use this kernel when
/// the accumulation work dominates the extra reads. An upfront bounds
/// check rejects out-of-range node ids before any worker starts, since
/// range partitioning would otherwise skip them silently.
///
/// # Examples
///
/// ```rust
/// use sonium_compute::explicit_scatter::{scatter_add_nodal, scatter_add_nodal_parallel};
///
/// let indices = vec![0, 1, 0, 1, 1];
/// let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
/// let mut serial = vec![0.0; 2];
/// let mut parallel = vec![0.0; 2];
/// scatter_add_nodal(&mut serial, &indices, &values);
/// scatter_add_nodal_parallel(&mut parallel, &indices, &values);
/// assert_eq!(serial, parallel);
/// ```
///
/// # Panics
///
/// Panics on layout mismatch (packed length not an exact multiple of
/// the index count, or nodal length not a multiple of the width) and
/// when a node id falls outside the nodal buffer.
pub fn scatter_add_nodal_parallel(target: &mut [f64], indices: &[usize], values: &[f64]) {
    let Some((width, nodes)) = packed_layout(target.len(), indices.len(), values.len()) else {
        return;
    };
    assert!(
        nodes > 0,
        "node index out of bounds: target holds no nodes but {} contributions were given",
        indices.len()
    );
    assert!(
        indices.iter().all(|&node| node < nodes),
        "node index out of bounds for {nodes} nodes"
    );
    let chunks = rayon::current_num_threads().max(1).min(nodes);
    let chunk_nodes = nodes.div_ceil(chunks);
    target
        .par_chunks_mut(chunk_nodes * width)
        .enumerate()
        .for_each(|(chunk, slice)| {
            let start = chunk * chunk_nodes;
            let end = (start + chunk_nodes).min(nodes);
            for (slot, &node) in indices.iter().enumerate() {
                if (start..end).contains(&node) {
                    let dst = (node - start) * width;
                    let src = slot * width;
                    let dst_row = &mut slice[dst..dst + width];
                    let src_row = &values[src..src + width];
                    for (slot, &value) in dst_row.iter_mut().zip(src_row.iter()) {
                        *slot += value;
                    }
                }
            }
        });
}

/// Assemble lumped masses from element volumes.
///
/// Adds `densities[e] * volumes[e] / node_count[e]` kilograms to every
/// node of element `e`, i.e. each element mass is split evenly over its
/// own nodes. Masses accumulate (`+=`) so successive element blocks can
/// assemble into one vector; zero the buffer first for a standalone
/// assembly. Inputs are SI: volumes in m^3, densities in kg/m^3, masses
/// in kg. The row-sum identity holds: the mass total equals the summed
/// element masses up to floating-point rounding.
///
/// # Examples
///
/// ```rust
/// use sonium_compute::explicit_scatter::{ElementTopology, assemble_lumped_mass};
///
/// let topology = ElementTopology::new(&[0, 1, 2, 3], &[0, 4]);
/// let mut masses = vec![0.0; 4];
/// assemble_lumped_mass(&mut masses, topology, &[8.0], &[2.0]);
/// assert_eq!(masses, vec![4.0; 4]);
/// ```
///
/// # Panics
///
/// Panics when `volumes` or `densities` does not hold one entry per
/// element, or when a node id falls outside `masses`.
pub fn assemble_lumped_mass(
    masses: &mut [f64],
    topology: ElementTopology<'_>,
    volumes: &[f64],
    densities: &[f64],
) {
    let elements = topology.element_count();
    assert!(
        volumes.len() == elements,
        "volumes length {} does not match element count {elements}",
        volumes.len()
    );
    assert!(
        densities.len() == elements,
        "densities length {} does not match element count {elements}",
        densities.len()
    );
    for element in 0..elements {
        let nodes = topology.nodes_of(element);
        let share = densities[element] * volumes[element] / nodes.len() as f64;
        for &node in nodes {
            masses[node] += share;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::explicit_state::ExplicitState;

    /// Deterministic fractional ramp (no RNG needed for exactness checks).
    fn ramp(len: usize) -> Vec<f64> {
        (0..len)
            .map(|i| ((i as f64) * 0.0123).sin() + ((i as f64) * 0.0456).cos())
            .collect()
    }

    /// Hex8 chain: element `e` owns nodes `(e + k) % nodes`, so every
    /// node is shared by 8 elements (the race case for scatter).
    fn chain_hex(nodes: usize, elements: usize) -> (Vec<usize>, Vec<usize>) {
        let mut connectivity = Vec::with_capacity(8 * elements);
        let mut offsets = Vec::with_capacity(elements + 1);
        for element in 0..elements {
            offsets.push(connectivity.len());
            for k in 0..8 {
                connectivity.push((element + k) % nodes);
            }
        }
        offsets.push(connectivity.len());
        (connectivity, offsets)
    }

    #[test]
    fn gather_scatter_round_trip_conserves_total() {
        // Integer-valued data keeps every sum exact.
        let field: Vec<f64> = (0..21).map(|i| i as f64).collect();
        let indices = vec![0, 3, 3, 6, 1, 0];
        let mut gathered = vec![0.0; 3 * indices.len()];
        gather_nodal(&field, &indices, &mut gathered);
        for (slot, &node) in indices.iter().enumerate() {
            assert_eq!(
                &gathered[3 * slot..3 * slot + 3],
                &field[3 * node..3 * node + 3]
            );
        }
        // Scattering the gathered rows back conserves the total exactly.
        let mut target = vec![0.0; field.len()];
        scatter_add_nodal(&mut target, &indices, &gathered);
        let want: f64 = gathered.iter().sum();
        let got: f64 = target.iter().sum();
        assert_eq!(got, want);
    }

    #[test]
    fn scatter_accumulates_shared_nodes() {
        let mut target = vec![0.0; 3];
        scatter_add_nodal(&mut target, &[2, 2, 2, 0], &[1.0, 2.0, 3.0, 10.0]);
        assert_eq!(target, vec![10.0, 0.0, 6.0]);
    }

    #[test]
    fn serial_parallel_scatter_agree_bit_exact() {
        // Odd node count and fractional values stress chunk edges and
        // summation order; agreement must still be bit-exact.
        let nodes = 37;
        let (connectivity, _) = chain_hex(nodes, 64);
        for width in [1, 3] {
            let values = ramp(width * connectivity.len());
            let nodal = if width == 1 { nodes } else { 3 * nodes };
            let mut serial = vec![0.0; nodal];
            let mut parallel = vec![0.0; nodal];
            let mut parallel_again = vec![0.0; nodal];
            scatter_add_nodal(&mut serial, &connectivity, &values);
            scatter_add_nodal_parallel(&mut parallel, &connectivity, &values);
            scatter_add_nodal_parallel(&mut parallel_again, &connectivity, &values);
            assert_eq!(
                parallel, serial,
                "width {width}: parallel differs from serial"
            );
            assert_eq!(
                parallel_again, serial,
                "width {width}: parallel run is not deterministic"
            );
        }
    }

    #[test]
    fn parallel_scatter_handles_more_threads_than_nodes() {
        // Fewer nodes than workers exercises single-node chunks.
        let nodes = 3;
        let (connectivity, _) = chain_hex(nodes, 25);
        let values = ramp(3 * connectivity.len());
        let mut serial = vec![0.0; 3 * nodes];
        let mut parallel = vec![0.0; 3 * nodes];
        scatter_add_nodal(&mut serial, &connectivity, &values);
        scatter_add_nodal_parallel(&mut parallel, &connectivity, &values);
        assert_eq!(parallel, serial);
    }

    #[test]
    fn empty_scatter_is_noop() {
        let mut serial = vec![1.0; 6];
        let mut parallel = vec![1.0; 6];
        scatter_add_nodal(&mut serial, &[], &[]);
        scatter_add_nodal_parallel(&mut parallel, &[], &[]);
        assert_eq!(serial, vec![1.0; 6]);
        assert_eq!(parallel, vec![1.0; 6]);
        let mut empty: Vec<f64> = Vec::new();
        scatter_add_nodal(&mut empty, &[], &[]);
        scatter_add_nodal_parallel(&mut empty, &[], &[]);
        assert!(empty.is_empty());
        let mut out: Vec<f64> = Vec::new();
        gather_nodal(&[], &[], &mut out);
        assert!(out.is_empty());
    }

    #[test]
    fn gather_packs_element_rows() {
        let field: Vec<f64> = (0..12).map(|i| i as f64 + 0.5).collect();
        let mut out = vec![0.0; 9];
        gather_nodal(&field, &[3, 0, 2], &mut out);
        assert_eq!(out, vec![9.5, 10.5, 11.5, 0.5, 1.5, 2.5, 6.5, 7.5, 8.5]);
    }

    #[test]
    fn lumped_mass_single_hex_uniform() {
        let connectivity: Vec<usize> = (0..8).collect();
        let topology = ElementTopology::new(&connectivity, &[0, 8]);
        let mut masses = vec![0.0; 8];
        assemble_lumped_mass(&mut masses, topology, &[8.0], &[2.0]);
        assert_eq!(masses, vec![2.0; 8]);
    }

    #[test]
    fn lumped_mass_scatters_share_evenly() {
        // 5 * 3 / 4 = 3.75 is exact in binary floating point.
        let topology = ElementTopology::new(&[0, 1, 2, 3], &[0, 4]);
        let mut masses = vec![0.0; 4];
        assemble_lumped_mass(&mut masses, topology, &[3.0], &[5.0]);
        assert_eq!(masses, vec![3.75; 4]);
    }

    #[test]
    fn lumped_mass_row_sum_matches_total() {
        // Mixed tri/tet/hex block with fractional volumes/densities.
        let connectivity = vec![0, 1, 2, 0, 1, 2, 3, 4, 5, 6, 7, 0, 3, 4, 8];
        let offsets = vec![0, 3, 7, 11, 15];
        let topology = ElementTopology::new(&connectivity, &offsets);
        let volumes = vec![1.1, 2.2, 3.3, 4.4];
        let densities = vec![7800.0, 2700.0, 1100.0, 7850.0];
        let mut masses = vec![0.0; 9];
        assemble_lumped_mass(&mut masses, topology, &volumes, &densities);
        assert!(masses.iter().all(|&mass| mass > 0.0));
        let total: f64 = masses.iter().sum();
        let want: f64 = volumes
            .iter()
            .zip(densities.iter())
            .map(|(&volume, &density)| density * volume)
            .sum();
        // Bound: under 200 accumulations at eps ~ 2.3e-16 keep the
        // relative error near 1e-13; 1e-12 carries full margin while a
        // dropped or duplicated element would err at the 1e-2 level.
        assert!(
            (total - want).abs() <= 1e-12 * want.abs().max(1.0),
            "row sum {total} differs from element total {want}"
        );
        // Assembly accumulates, so a second pass doubles the masses.
        assemble_lumped_mass(&mut masses, topology, &volumes, &densities);
        let doubled: f64 = masses.iter().sum();
        assert!(
            (doubled - 2.0 * total).abs() <= 1e-12 * want.abs().max(1.0),
            "second assembly did not accumulate: {doubled} vs {}",
            2.0 * total
        );
    }

    #[test]
    fn explicit_state_interop() {
        let mut state = ExplicitState::new(3);
        state.zero_forces();
        let forces = vec![1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0];
        scatter_add_nodal(&mut state.forces, &[0, 1, 2], &forces);
        assert_eq!(state.forces, forces);
        let topology = ElementTopology::new(&[0, 1, 2], &[0, 3]);
        assemble_lumped_mass(&mut state.masses, topology, &[6.0], &[10.0]);
        assert_eq!(state.masses, vec![20.0; 3]);
        assert!(state.is_consistent());
    }

    #[test]
    fn topology_accessors() {
        let topology = ElementTopology::new(&[0, 1, 2, 3, 4], &[0, 2, 5]);
        assert_eq!(topology.element_count(), 2);
        assert_eq!(topology.nodes_of(0), &[0, 1]);
        assert_eq!(topology.nodes_of(1), &[2, 3, 4]);
    }

    #[test]
    #[should_panic(expected = "packed length")]
    fn gather_rejects_bad_out_length() {
        let mut out = vec![0.0; 5];
        gather_nodal(&[0.0; 6], &[0, 1], &mut out);
    }

    #[test]
    #[should_panic(expected = "nodal length")]
    fn gather_rejects_bad_field_length() {
        let mut out = vec![0.0; 4];
        gather_nodal(&[0.0; 5], &[0, 1], &mut out);
    }

    #[test]
    #[should_panic(expected = "packed length")]
    fn scatter_rejects_bad_values_length() {
        let mut target = vec![0.0; 6];
        scatter_add_nodal(&mut target, &[0, 1], &[1.0; 5]);
    }

    #[test]
    #[should_panic(expected = "packed length")]
    fn parallel_scatter_rejects_bad_values_length() {
        let mut target = vec![0.0; 6];
        scatter_add_nodal_parallel(&mut target, &[0, 1], &[1.0; 5]);
    }

    #[test]
    #[should_panic(expected = "out of range")]
    fn serial_scatter_rejects_bad_node_id() {
        let mut target = vec![0.0; 6];
        scatter_add_nodal(&mut target, &[0, 5], &[1.0; 6]);
    }

    #[test]
    #[should_panic(expected = "out of bounds")]
    fn parallel_scatter_rejects_bad_node_id() {
        let mut target = vec![0.0; 6];
        scatter_add_nodal_parallel(&mut target, &[0, 5], &[1.0; 6]);
    }

    #[test]
    #[should_panic(expected = "holds no nodes")]
    fn parallel_scatter_rejects_empty_target() {
        let mut target: Vec<f64> = Vec::new();
        scatter_add_nodal_parallel(&mut target, &[0], &[1.0]);
    }

    #[test]
    #[should_panic(expected = "empty slice")]
    fn topology_rejects_empty_offsets() {
        let _ = ElementTopology::new(&[0], &[]);
    }

    #[test]
    #[should_panic(expected = "strictly increase")]
    fn topology_rejects_empty_element() {
        let _ = ElementTopology::new(&[0, 1], &[0, 2, 2]);
    }

    #[test]
    #[should_panic(expected = "volumes length")]
    fn lumped_mass_rejects_bad_volumes_length() {
        let topology = ElementTopology::new(&[0, 1], &[0, 2]);
        let mut masses = vec![0.0; 2];
        assemble_lumped_mass(&mut masses, topology, &[], &[1.0]);
    }
}
