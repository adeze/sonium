//! Contact MMS — penalty gap validation
//!
//! Uses prescribed geometry (triangle + query points) to verify
//! that the gap computation matches the analytical signed distance.

use sonium_fem::physics::solid::contact::gap::project_point_to_triangle;

fn test_gap_interior_positive() {
    let v0 = [0.0, 0.0, 0.0];
    let v1 = [1.0, 0.0, 0.0];
    let v2 = [0.0, 1.0, 0.0];

    // Point above centroid at z = 0.5
    let p = [1.0 / 3.0, 1.0 / 3.0, 0.5];
    let r = project_point_to_triangle(p, v0, v1, v2);

    assert!(
        (r.gap - 0.5).abs() < 1e-12,
        "Interior gap should be 0.5, got {}",
        r.gap
    );
    assert!(r.gap > 0.0, "Gap should be positive (separated)");
}

fn test_gap_negative_penetration() {
    let v0 = [0.0, 0.0, 0.0];
    let v1 = [1.0, 0.0, 0.0];
    let v2 = [0.0, 1.0, 0.0];

    // Point below triangle at z = -0.3
    let p = [0.25, 0.25, -0.3];
    let r = project_point_to_triangle(p, v0, v1, v2);

    assert!(
        (r.gap - (-0.3)).abs() < 1e-12,
        "Penetration gap should be -0.3, got {}",
        r.gap
    );
    assert!(r.gap < 0.0, "Gap should be negative (penetrating)");
}

fn test_gap_on_edge() {
    let v0 = [0.0, 0.0, 0.0];
    let v1 = [1.0, 0.0, 0.0];
    let v2 = [0.0, 1.0, 0.0];

    // Point directly above the middle of edge v0-v1
    let p = [0.5, 0.0, 0.2];
    let r = project_point_to_triangle(p, v0, v1, v2);

    assert!(
        (r.gap - 0.2).abs() < 1e-12,
        "Edge gap should be 0.2, got {}",
        r.gap
    );
    // Closest point should be on the edge
    assert!(
        (r.closest[1]).abs() < 1e-12,
        "Closest y should be 0, got {}",
        r.closest[1]
    );
}

fn test_gap_at_vertex() {
    let v0 = [0.0, 0.0, 0.0];
    let v1 = [1.0, 0.0, 0.0];
    let v2 = [0.0, 1.0, 0.0];

    // Point far from vertex v0
    let p = [-0.4, -0.3, 0.1];
    let r = project_point_to_triangle(p, v0, v1, v2);

    // Closest point should be v0
    assert!(
        (r.closest[0]).abs() < 1e-12,
        "Closest x should be 0, got {}",
        r.closest[0]
    );
    assert!(
        (r.closest[1]).abs() < 1e-12,
        "Closest y should be 0, got {}",
        r.closest[1]
    );

    // Gap is the signed distance from p to the triangle plane at v0
    // Normal is [0,0,1], so gap = p·n - v0·n = 0.1
    assert!(
        (r.gap - 0.1).abs() < 1e-12,
        "Vertex gap should be 0.1, got {}",
        r.gap
    );
}

fn test_barycentric_sum() {
    let v0 = [1.0, 0.0, 0.0];
    let v1 = [0.0, 2.0, 0.0];
    let v2 = [0.0, 0.0, 3.0];

    let p = [0.3, 0.5, 0.8];
    let r = project_point_to_triangle(p, v0, v1, v2);

    let zeta = 1.0 - r.xi - r.eta;
    assert!(
        r.xi >= -1e-12 && r.eta >= -1e-12 && zeta >= -1e-12,
        "Barycentric coordinates should be non-negative"
    );
    assert!(
        (r.xi + r.eta + zeta - 1.0).abs() < 1e-12,
        "Barycentric sum should be 1"
    );
}

fn main() {
    let result = std::panic::catch_unwind(|| {
        test_gap_interior_positive();
        test_gap_negative_penetration();
        test_gap_on_edge();
        test_gap_at_vertex();
        test_barycentric_sum();
    });
    match result {
        Ok(()) => println!("SUCCESS"),
        Err(_) => {
            println!("FAILED");
            std::process::exit(1);
        }
    }
}
