use super::{tet_edges, tet_faces};
use std::collections::{HashMap, HashSet};
use std::hash::Hash;

#[derive(Clone)]
pub(super) struct RecoveryTopology {
    pub(super) vertices: HashMap<usize, Vec<usize>>,
    pub(super) edges: HashMap<[usize; 2], Vec<usize>>,
    pub(super) faces: HashMap<[usize; 3], Vec<usize>>,
}

impl RecoveryTopology {
    pub(super) fn from_tets(tets: &[[usize; 4]]) -> Self {
        let mut topology = Self {
            vertices: HashMap::new(),
            edges: HashMap::new(),
            faces: HashMap::new(),
        };
        for (idx, tet) in tets.iter().copied().enumerate() {
            topology.add_tet(idx, tet);
        }
        topology
    }

    pub(super) fn has_edge(&self, edge: [usize; 2]) -> bool {
        self.edges.contains_key(&edge)
    }

    pub(super) fn has_face(&self, face: [usize; 3]) -> bool {
        self.faces.contains_key(&face)
    }

    pub(super) fn edge_set(&self) -> HashSet<[usize; 2]> {
        self.edges.keys().copied().collect()
    }

    pub(super) fn face_set(&self) -> HashSet<[usize; 3]> {
        self.faces.keys().copied().collect()
    }

    pub(super) fn edge_owners(&self, edge: [usize; 2]) -> &[usize] {
        self.edges.get(&edge).map(Vec::as_slice).unwrap_or(&[])
    }

    pub(super) fn vertex_owners(&self, vertex: usize) -> &[usize] {
        self.vertices.get(&vertex).map(Vec::as_slice).unwrap_or(&[])
    }

    pub(super) fn add_tet(&mut self, idx: usize, tet: [usize; 4]) {
        for vertex in tet {
            self.vertices.entry(vertex).or_default().push(idx);
        }
        for edge in tet_edges(tet) {
            self.edges.entry(edge).or_default().push(idx);
        }
        for face in tet_faces(tet) {
            self.faces.entry(face).or_default().push(idx);
        }
    }

    pub(super) fn remove_tet(&mut self, idx: usize, tet: [usize; 4]) {
        for vertex in tet {
            remove_incidence(&mut self.vertices, vertex, idx);
        }
        for edge in tet_edges(tet) {
            remove_incidence(&mut self.edges, edge, idx);
        }
        for face in tet_faces(tet) {
            remove_incidence(&mut self.faces, face, idx);
        }
    }

    pub(super) fn remap_tet_index(&mut self, old_idx: usize, new_idx: usize, tet: [usize; 4]) {
        for vertex in tet {
            remap_incidence(&mut self.vertices, vertex, old_idx, new_idx);
        }
        for edge in tet_edges(tet) {
            remap_incidence(&mut self.edges, edge, old_idx, new_idx);
        }
        for face in tet_faces(tet) {
            remap_incidence(&mut self.faces, face, old_idx, new_idx);
        }
    }
}

fn remove_incidence<K: Copy + Eq + Hash>(map: &mut HashMap<K, Vec<usize>>, key: K, idx: usize) {
    let Some(owners) = map.get_mut(&key) else {
        return;
    };
    owners.retain(|owner| *owner != idx);
    if owners.is_empty() {
        map.remove(&key);
    }
}

fn remap_incidence<K: Copy + Eq + Hash>(
    map: &mut HashMap<K, Vec<usize>>,
    key: K,
    old_idx: usize,
    new_idx: usize,
) {
    if let Some(owners) = map.get_mut(&key) {
        for owner in owners {
            if *owner == old_idx {
                *owner = new_idx;
            }
        }
    }
}
