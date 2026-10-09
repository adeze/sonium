//! Contact detection via AABB bounding volume hierarchy
//!
//! Builds an axis-aligned bounding box tree over the deformed surface triangles
//! of a 3D mesh. Supports self-contact queries by filtering out topologically
//! adjacent triangles.

/// Axis-aligned bounding box.
#[derive(Debug, Clone, Copy)]
pub struct Aabb {
    pub min: [f64; 3],
    pub max: [f64; 3],
}

impl Aabb {
    /// Create from a set of points.
    pub fn from_points(points: &[[f64; 3]]) -> Self {
        let mut min = [f64::MAX; 3];
        let mut max = [f64::MIN; 3];
        for p in points {
            for d in 0..3 {
                min[d] = min[d].min(p[d]);
                max[d] = max[d].max(p[d]);
            }
        }
        Self { min, max }
    }

    /// Expand the box by `margin` in all directions.
    pub fn inflate(&self, margin: f64) -> Self {
        Self {
            min: [
                self.min[0] - margin,
                self.min[1] - margin,
                self.min[2] - margin,
            ],
            max: [
                self.max[0] + margin,
                self.max[1] + margin,
                self.max[2] + margin,
            ],
        }
    }

    /// Merge two AABBs into the smallest enclosing AABB.
    pub fn merge(&self, other: &Aabb) -> Self {
        Self {
            min: [
                self.min[0].min(other.min[0]),
                self.min[1].min(other.min[1]),
                self.min[2].min(other.min[2]),
            ],
            max: [
                self.max[0].max(other.max[0]),
                self.max[1].max(other.max[1]),
                self.max[2].max(other.max[2]),
            ],
        }
    }

    /// Test overlap with another AABB.
    pub fn intersects(&self, other: &Aabb) -> bool {
        self.min[0] <= other.max[0]
            && self.max[0] >= other.min[0]
            && self.min[1] <= other.max[1]
            && self.max[1] >= other.min[1]
            && self.min[2] <= other.max[2]
            && self.max[2] >= other.min[2]
    }

    /// Longest axis (0=x, 1=y, 2=z).
    fn longest_axis(&self) -> usize {
        let dx = self.max[0] - self.min[0];
        let dy = self.max[1] - self.min[1];
        let dz = self.max[2] - self.min[2];
        if dx >= dy && dx >= dz {
            0
        } else if dy >= dz {
            1
        } else {
            2
        }
    }
}

/// BVH node: either a leaf (single triangle) or an internal node with two children.
#[derive(Debug)]
enum BvhNode {
    Leaf {
        bbox: Aabb,
        tri_idx: usize,
    },
    Internal {
        bbox: Aabb,
        left: usize, // index into AabbTree::nodes
        right: usize,
    },
}

impl BvhNode {
    fn bbox(&self) -> &Aabb {
        match self {
            BvhNode::Leaf { bbox, .. } => bbox,
            BvhNode::Internal { bbox, .. } => bbox,
        }
    }
}

/// AABB bounding volume hierarchy over surface triangles.
#[derive(Debug)]
pub struct AabbTree {
    nodes: Vec<BvhNode>,
    root: usize,
}

impl AabbTree {
    /// Build the BVH from triangle AABBs.
    ///
    /// `tri_bboxes` is indexed by triangle index.
    /// `tri_centroids` provides the centroid of each triangle for spatial sorting.
    pub fn build(tri_bboxes: &[Aabb], tri_centroids: &[[f64; 3]]) -> Self {
        let n = tri_bboxes.len();
        assert!(n > 0);

        let mut indices: Vec<usize> = (0..n).collect();
        let mut nodes = Vec::with_capacity(2 * n);

        let root = Self::build_recursive(&mut nodes, tri_bboxes, tri_centroids, &mut indices);

        Self { nodes, root }
    }

    fn build_recursive(
        nodes: &mut Vec<BvhNode>,
        bboxes: &[Aabb],
        centroids: &[[f64; 3]],
        indices: &mut [usize],
    ) -> usize {
        if indices.len() == 1 {
            let idx = nodes.len();
            nodes.push(BvhNode::Leaf {
                bbox: bboxes[indices[0]],
                tri_idx: indices[0],
            });
            return idx;
        }

        // Compute enclosing AABB of all triangles in this subset
        let mut enclosing = bboxes[indices[0]];
        for &i in indices.iter().skip(1) {
            enclosing = enclosing.merge(&bboxes[i]);
        }

        // Split along longest axis at median centroid
        let axis = enclosing.longest_axis();
        indices.sort_unstable_by(|&a, &b| {
            centroids[a][axis]
                .partial_cmp(&centroids[b][axis])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let mid = indices.len() / 2;
        let (left_slice, right_slice) = indices.split_at_mut(mid);

        let left = Self::build_recursive(nodes, bboxes, centroids, left_slice);
        let right = Self::build_recursive(nodes, bboxes, centroids, right_slice);

        let bbox = nodes[left].bbox().merge(nodes[right].bbox());
        let idx = nodes.len();
        nodes.push(BvhNode::Internal { bbox, left, right });
        idx
    }

    /// Query all triangle indices whose AABB intersects `query_box`.
    pub fn query(&self, query_box: &Aabb, results: &mut Vec<usize>) {
        self.query_recursive(self.root, query_box, results);
    }

    fn query_recursive(&self, node_idx: usize, query_box: &Aabb, results: &mut Vec<usize>) {
        match &self.nodes[node_idx] {
            BvhNode::Leaf { bbox, tri_idx } => {
                if bbox.intersects(query_box) {
                    results.push(*tri_idx);
                }
            }
            BvhNode::Internal { bbox, left, right } => {
                if bbox.intersects(query_box) {
                    self.query_recursive(*left, query_box, results);
                    self.query_recursive(*right, query_box, results);
                }
            }
        }
    }
}

/// A surface triangle with parent mesh node indices.
#[derive(Debug, Clone)]
pub struct SurfaceTriangle {
    /// Parent mesh node indices (3 nodes).
    pub nodes: [usize; 3],
    /// Set of node indices for adjacency checks (includes nodes from neighboring triangles).
    /// Used to filter self-contact: skip pairs where the query node belongs to this set.
    pub adjacent_nodes: Vec<usize>,
}

/// Extracted contact surface from a 3D mesh.
#[derive(Debug)]
pub struct ContactSurface {
    /// Surface triangles with parent node indices.
    pub triangles: Vec<SurfaceTriangle>,
    /// Unique surface node indices (into parent mesh).
    pub node_indices: Vec<usize>,
    /// Map from parent node index to surface node index.
    pub parent_to_surface: Vec<Option<usize>>,
}

impl ContactSurface {
    /// Extract contact surface from mesh boundary faces.
    ///
    /// `nodes` are the mesh node coordinates (reference configuration).
    /// `boundary_faces` are the boundary face node lists.
    /// `n_parent_nodes` is the total number of nodes in the parent mesh.
    pub fn from_boundary_faces(boundary_faces: &[[usize; 3]], n_parent_nodes: usize) -> Self {
        // Collect unique surface nodes
        let mut parent_to_surface = vec![None; n_parent_nodes];
        let mut node_indices = Vec::new();

        for face in boundary_faces {
            for &n in face {
                if parent_to_surface[n].is_none() {
                    parent_to_surface[n] = Some(node_indices.len());
                    node_indices.push(n);
                }
            }
        }

        // Build triangles with adjacency info
        // Two triangles are adjacent if they share at least one node
        let mut triangles: Vec<SurfaceTriangle> = boundary_faces
            .iter()
            .map(|&nodes| SurfaceTriangle {
                nodes,
                adjacent_nodes: Vec::new(),
            })
            .collect();

        // Build node-to-triangle connectivity for adjacency
        let mut node_to_tris: Vec<Vec<usize>> = vec![Vec::new(); n_parent_nodes];
        for (tri_idx, tri) in triangles.iter().enumerate() {
            for &n in &tri.nodes {
                node_to_tris[n].push(tri_idx);
            }
        }

        // For each triangle, collect all nodes from adjacent triangles
        for tri_idx in 0..triangles.len() {
            let mut adj_nodes = std::collections::HashSet::new();
            // Add own nodes
            for &n in &triangles[tri_idx].nodes {
                adj_nodes.insert(n);
            }
            // Add nodes from triangles that share a node with this one
            for &n in &triangles[tri_idx].nodes {
                for &neighbor_tri in &node_to_tris[n] {
                    for &nn in &triangles[neighbor_tri].nodes {
                        adj_nodes.insert(nn);
                    }
                }
            }
            triangles[tri_idx].adjacent_nodes = adj_nodes.into_iter().collect();
        }

        Self {
            triangles,
            node_indices,
            parent_to_surface,
        }
    }

    /// Build the AABB tree from deformed node positions.
    ///
    /// `deformed_coords` is the full parent mesh coordinate array (3 * n_parent_nodes).
    pub fn build_bvh(&self, deformed_coords: &[f64], margin: f64) -> AabbTree {
        let bboxes: Vec<Aabb> = self
            .triangles
            .iter()
            .map(|tri| {
                let pts: Vec<[f64; 3]> = tri
                    .nodes
                    .iter()
                    .map(|&n| {
                        [
                            deformed_coords[3 * n],
                            deformed_coords[3 * n + 1],
                            deformed_coords[3 * n + 2],
                        ]
                    })
                    .collect();
                Aabb::from_points(&pts).inflate(margin)
            })
            .collect();

        let centroids: Vec<[f64; 3]> = self
            .triangles
            .iter()
            .map(|tri| {
                let mut c = [0.0; 3];
                for &n in &tri.nodes {
                    c[0] += deformed_coords[3 * n];
                    c[1] += deformed_coords[3 * n + 1];
                    c[2] += deformed_coords[3 * n + 2];
                }
                c[0] /= 3.0;
                c[1] /= 3.0;
                c[2] /= 3.0;
                c
            })
            .collect();

        AabbTree::build(&bboxes, &centroids)
    }

    /// Find candidate contact pairs (node_index, triangle_index) for self-contact.
    ///
    /// For each surface node, queries the BVH for nearby triangles, filtering
    /// out triangles adjacent to the node (topological filter).
    ///
    /// `deformed_coords`: full parent mesh coords (3 * n_parent_nodes).
    /// `search_radius`: how far from each node to search for candidate triangles.
    pub fn find_candidates(
        &self,
        deformed_coords: &[f64],
        bvh: &AabbTree,
        search_radius: f64,
    ) -> Vec<(usize, usize)> {
        self.find_candidates_for(&self.node_indices, deformed_coords, bvh, search_radius)
    }

    /// Candidate pairs for an explicit slave-node list against this
    /// surface's triangles (one-sided two-body contact: slaves live on
    /// the opposite body, so each physical contact point pairs once).
    pub fn find_candidates_for(
        &self,
        slave_nodes: &[usize],
        deformed_coords: &[f64],
        bvh: &AabbTree,
        search_radius: f64,
    ) -> Vec<(usize, usize)> {
        let mut candidates = Vec::new();
        let mut bvh_hits = Vec::new();

        for &node_idx in slave_nodes {
            let x = [
                deformed_coords[3 * node_idx],
                deformed_coords[3 * node_idx + 1],
                deformed_coords[3 * node_idx + 2],
            ];

            let query_box = Aabb {
                min: [
                    x[0] - search_radius,
                    x[1] - search_radius,
                    x[2] - search_radius,
                ],
                max: [
                    x[0] + search_radius,
                    x[1] + search_radius,
                    x[2] + search_radius,
                ],
            };

            bvh_hits.clear();
            bvh.query(&query_box, &mut bvh_hits);

            for &tri_idx in &bvh_hits {
                // Topological filter: skip if node belongs to this triangle or its neighbors
                if self.triangles[tri_idx].adjacent_nodes.contains(&node_idx) {
                    continue;
                }
                candidates.push((node_idx, tri_idx));
            }
        }

        candidates
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_aabb_intersects() {
        let a = Aabb {
            min: [0.0, 0.0, 0.0],
            max: [1.0, 1.0, 1.0],
        };
        let b = Aabb {
            min: [0.5, 0.5, 0.5],
            max: [1.5, 1.5, 1.5],
        };
        let c = Aabb {
            min: [2.0, 2.0, 2.0],
            max: [3.0, 3.0, 3.0],
        };
        assert!(a.intersects(&b));
        assert!(!a.intersects(&c));
    }

    #[test]
    fn test_aabb_inflate() {
        let a = Aabb {
            min: [1.0, 1.0, 1.0],
            max: [2.0, 2.0, 2.0],
        };
        let b = a.inflate(0.5);
        assert!((b.min[0] - 0.5).abs() < 1e-14);
        assert!((b.max[0] - 2.5).abs() < 1e-14);
    }

    #[test]
    fn test_bvh_query() {
        // 4 triangles forming a square: two on left, two on right
        let bboxes = vec![
            Aabb {
                min: [0.0, 0.0, 0.0],
                max: [1.0, 1.0, 0.0],
            },
            Aabb {
                min: [1.0, 0.0, 0.0],
                max: [2.0, 1.0, 0.0],
            },
            Aabb {
                min: [0.0, 1.0, 0.0],
                max: [1.0, 2.0, 0.0],
            },
            Aabb {
                min: [1.0, 1.0, 0.0],
                max: [2.0, 2.0, 0.0],
            },
        ];
        let centroids = vec![
            [0.5, 0.5, 0.0],
            [1.5, 0.5, 0.0],
            [0.5, 1.5, 0.0],
            [1.5, 1.5, 0.0],
        ];

        let tree = AabbTree::build(&bboxes, &centroids);

        // Query box overlapping left half
        let query = Aabb {
            min: [-0.1, -0.1, -1.0],
            max: [1.1, 2.1, 1.0],
        };
        let mut results = Vec::new();
        tree.query(&query, &mut results);
        results.sort();

        // Should hit triangles 0, 1, 2, 3 (1 is at x=1.0 boundary, overlaps with 1.1)
        assert!(results.contains(&0));
        assert!(results.contains(&2));
    }

    #[test]
    fn test_contact_surface_adjacency_filter() {
        // Two triangles sharing an edge (nodes 0,1 shared)
        //   tri 0: [0, 1, 2]
        //   tri 1: [0, 1, 3]
        // Node 2 is only in tri 0, node 3 only in tri 1
        let faces = vec![[0, 1, 2], [0, 1, 3]];
        let surface = ContactSurface::from_boundary_faces(&faces, 4);

        // Node 2 should be adjacent to tri 0 (it's IN tri 0)
        assert!(surface.triangles[0].adjacent_nodes.contains(&2));
        // Node 2 should also be adjacent to tri 1 via the shared edge
        // (tri 0 and tri 1 share nodes 0,1, so tri 1's adjacency includes tri 0's nodes)
        assert!(surface.triangles[1].adjacent_nodes.contains(&2));

        // Node 3 should be adjacent to tri 0 (via shared edge with tri 1)
        assert!(surface.triangles[0].adjacent_nodes.contains(&3));
    }
}
