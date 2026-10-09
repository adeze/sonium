//! FEM-facing mesh-motion contract.
//!
//! Motion ownership lives in `sonium-mesher`: the FEM layer re-exports that
//! checked, transactional API so simulation callers and mesh transformations
//! use the same admission, orientation, and quality-barrier policy.

pub use sonium_mesher::mesh::motion::{
    HarmonicMotionOptions, HarmonicMotionReport, MeshMotion, MeshMotionError, MeshMotionReport,
};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::{ElementType, Mesh, Point};

    fn triangle_mesh() -> Mesh {
        let mut mesh = Mesh::new(2);
        mesh.add_node(Point::new_2d(0.0, 0.0));
        mesh.add_node(Point::new_2d(1.0, 0.0));
        mesh.add_node(Point::new_2d(0.0, 1.0));
        mesh.add_element(ElementType::Triangle, vec![0, 1, 2]);
        mesh
    }

    #[test]
    fn valid_motion_updates_mesh_and_reports_displacement() {
        let mut mesh = triangle_mesh();
        let motion = MeshMotion::new(vec![[0.1, 0.0, 0.0]; 3]).expect("finite motion");
        let report = motion.apply_checked(&mut mesh).expect("valid motion");
        assert_eq!(report.moved_nodes, 3);
        assert!((report.max_displacement - 0.1).abs() < 1e-12);
        assert!((mesh.nodes[0].x - 0.1).abs() < 1e-12);
    }

    #[test]
    fn inverted_motion_is_transactional() {
        let mut mesh = triangle_mesh();
        let original_nodes = mesh.nodes.clone();
        let motion = MeshMotion::new(vec![[0.0, 0.0, 0.0], [-2.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
            .expect("finite motion");
        assert!(matches!(
            motion.apply_checked(&mut mesh),
            Err(MeshMotionError::InvalidElements(_))
        ));
        assert_eq!(mesh.nodes, original_nodes);
    }

    #[test]
    fn malformed_motion_is_rejected_before_mesh_mutation() {
        let mut mesh = triangle_mesh();
        let original_nodes = mesh.nodes.clone();
        let motion = MeshMotion::new(vec![[0.0, 0.0, 0.0]; 2]).expect("field itself is finite");
        assert!(matches!(
            motion.apply_checked(&mut mesh),
            Err(MeshMotionError::NodeCount { .. })
        ));
        assert_eq!(mesh.nodes, original_nodes);
        assert!(matches!(
            MeshMotion::new(vec![[f64::NAN, 0.0, 0.0]; 3]),
            Err(MeshMotionError::NonFinite { node: 0 })
        ));
    }
}
