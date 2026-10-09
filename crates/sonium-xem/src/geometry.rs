//! Serializable room geometry used by exchange artifacts.
//!
//! Surface-mesh generation deliberately lives in `sonium-mesher`.  Consumers
//! that need it should import [`sonium_mesher::RoomSurfaceMeshing`]; keeping
//! this module data-only prevents an exchange artifact from selecting a
//! solver-specific discretisation algorithm.

use crate::types::Point3D;
use serde::{Deserialize, Serialize};

/// Room geometry types.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RoomGeometry {
    /// Rectangular (shoebox) room.
    Rectangular(RectangularRoom),
    /// L-shaped room.
    LShaped(LShapedRoom),
}

impl RoomGeometry {
    /// Get the edges of the room geometry for visualization.
    pub fn get_edges(&self) -> Vec<(Point3D, Point3D)> {
        match self {
            Self::Rectangular(room) => room.get_edges(),
            Self::LShaped(room) => room.get_edges(),
        }
    }

    /// Get room dimensions (width, depth, height).
    pub fn dimensions(&self) -> (f64, f64, f64) {
        match self {
            Self::Rectangular(r) => (r.width, r.depth, r.height),
            Self::LShaped(r) => (r.width1.max(r.width2), r.depth1 + r.depth2, r.height),
        }
    }

    /// Get room volume in cubic meters.
    pub fn volume(&self) -> f64 {
        match self {
            Self::Rectangular(r) => r.width * r.depth * r.height,
            Self::LShaped(r) => r.width1 * r.depth1 * r.height + r.width2 * r.depth2 * r.height,
        }
    }
}

/// Rectangular room defined by dimensions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RectangularRoom {
    /// Room width (x dimension).
    pub width: f64,
    /// Room depth (y dimension).
    pub depth: f64,
    /// Room height (z dimension).
    pub height: f64,
}

impl RectangularRoom {
    /// Create a rectangular room with the supplied dimensions.
    pub fn new(width: f64, depth: f64, height: f64) -> Self {
        Self {
            width,
            depth,
            height,
        }
    }

    /// Get room edges for visualization.
    pub fn get_edges(&self) -> Vec<(Point3D, Point3D)> {
        prism_edges(
            &[
                Point3D::new(0.0, 0.0, 0.0),
                Point3D::new(self.width, 0.0, 0.0),
                Point3D::new(self.width, self.depth, 0.0),
                Point3D::new(0.0, self.depth, 0.0),
            ],
            self.height,
        )
    }
}

/// L-shaped room defined by two rectangular sections.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LShapedRoom {
    /// Main section width (x dimension).
    pub width1: f64,
    /// Main section depth (y dimension).
    pub depth1: f64,
    /// Extension width (x dimension), typically less than `width1`.
    pub width2: f64,
    /// Extension depth (y dimension).
    pub depth2: f64,
    /// Common height for both sections (z dimension).
    pub height: f64,
}

impl LShapedRoom {
    /// Create an L-shaped room with the supplied dimensions.
    pub fn new(width1: f64, depth1: f64, width2: f64, depth2: f64, height: f64) -> Self {
        Self {
            width1,
            depth1,
            width2,
            depth2,
            height,
        }
    }

    /// Get room edges for visualization.
    pub fn get_edges(&self) -> Vec<(Point3D, Point3D)> {
        let depth = self.depth1 + self.depth2;
        prism_edges(
            &[
                Point3D::new(0.0, 0.0, 0.0),
                Point3D::new(self.width1, 0.0, 0.0),
                Point3D::new(self.width1, self.depth1, 0.0),
                Point3D::new(self.width2, self.depth1, 0.0),
                Point3D::new(self.width2, depth, 0.0),
                Point3D::new(0.0, depth, 0.0),
            ],
            self.height,
        )
    }
}

fn prism_edges(floor: &[Point3D], height: f64) -> Vec<(Point3D, Point3D)> {
    let ceiling: Vec<_> = floor
        .iter()
        .map(|p| Point3D::new(p.x, p.y, height))
        .collect();
    let mut edges = Vec::with_capacity(floor.len() * 3);
    for index in 0..floor.len() {
        let next = (index + 1) % floor.len();
        edges.push((floor[index], floor[next]));
        edges.push((ceiling[index], ceiling[next]));
        edges.push((floor[index], ceiling[index]));
    }
    edges
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exchange_geometry_retains_dimensions_volume_and_edges_without_meshing() {
        let room = RoomGeometry::LShaped(LShapedRoom::new(5.0, 4.0, 3.0, 3.0, 2.5));
        assert_eq!(room.dimensions(), (5.0, 7.0, 2.5));
        assert_eq!(room.volume(), 72.5);
        assert_eq!(room.get_edges().len(), 18);
    }
}
