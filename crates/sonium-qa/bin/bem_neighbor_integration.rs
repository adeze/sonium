//! Neighbor Integration Test
//!
//! Test if using higher integration accuracy for neighbor elements improves E[1].
//! Hypothesis: Edge-adjacent elements need special treatment.

fn test_neighbor_integration() {
    use num_complex::Complex64;
    use sonium_bem::core::integration::{
        regular_integration, regular_integration_fixed_order, singular_integration,
    };
    use sonium_bem::core::mesh::generators::generate_icosphere_mesh;
    use sonium_bem::core::types::PhysicsParams;
    use sonium_qa::relative_error;
    use std::collections::HashSet;
    use std::f64::consts::PI;

    let radius = 0.1;
    let speed_of_sound = 343.0;
    let density = 1.21;
    let ka_target = 0.2;
    let frequency = ka_target * speed_of_sound / (2.0 * PI * radius);

    println!("=== Neighbor Integration Test ===\n");
    println!("Testing if higher quadrature for neighbor elements improves E[1]\n");

    for subdivisions in [2, 3] {
        let mesh = generate_icosphere_mesh(radius, subdivisions);
        let physics = PhysicsParams::new(frequency, speed_of_sound, density, false);

        println!(
            "--- Icosphere subdivisions={} ({} elements) ---",
            subdivisions,
            mesh.elements.len()
        );

        // Find neighbor elements for source element 0
        let source_idx = 0;
        let source_elem = &mesh.elements[source_idx];
        let source_point = &source_elem.center;
        let source_normal = &source_elem.normal;
        let source_nodes: HashSet<usize> = source_elem.connectivity.iter().copied().collect();

        // Find elements sharing at least one node
        let mut neighbors: HashSet<usize> = HashSet::new();
        for (j, elem) in mesh.elements.iter().enumerate() {
            if j == source_idx {
                continue;
            }
            for &node in &elem.connectivity {
                if source_nodes.contains(&node) {
                    neighbors.insert(j);
                    break;
                }
            }
        }

        println!(
            "  Source element {} has {} neighbor elements",
            source_idx,
            neighbors.len()
        );

        // Compute E integrals with STANDARD integration
        let source_coords = mesh.element_nodes(source_elem);
        let self_result = singular_integration(
            source_point,
            source_normal,
            &source_coords,
            source_elem.element_type,
            &physics,
            None,
            0,
            false,
        );
        let self_e = self_result.d2g_dnxdny_integral;

        let mut offdiag_e_standard = Complex64::new(0.0, 0.0);
        let mut neighbor_e_standard = Complex64::new(0.0, 0.0);
        let mut far_e_standard = Complex64::new(0.0, 0.0);

        for (j, field_elem) in mesh.elements.iter().enumerate() {
            if j == source_idx {
                continue;
            }
            let element_coords = mesh.element_nodes(field_elem);
            let result = regular_integration(
                source_point,
                source_normal,
                &element_coords,
                field_elem.element_type,
                field_elem.area,
                &physics,
                None,
                0,
                false,
            );
            offdiag_e_standard += result.d2g_dnxdny_integral;
            if neighbors.contains(&j) {
                neighbor_e_standard += result.d2g_dnxdny_integral;
            } else {
                far_e_standard += result.d2g_dnxdny_integral;
            }
        }

        let e1_standard = self_e + offdiag_e_standard;

        println!("  Standard integration:");
        println!("    Self E: {:.4}", self_e.re);
        println!(
            "    Neighbor E sum: {:.4} ({} elements)",
            neighbor_e_standard.re,
            neighbors.len()
        );
        println!("    Far E sum: {:.4}", far_e_standard.re);
        println!("    E[1] closure diagnostic = {:.4}", e1_standard.re);

        // Compute E integrals with HIGH-ORDER integration for neighbors
        let mut offdiag_e_high = Complex64::new(0.0, 0.0);
        let mut neighbor_e_high = Complex64::new(0.0, 0.0);

        for (j, field_elem) in mesh.elements.iter().enumerate() {
            if j == source_idx {
                continue;
            }
            let element_coords = mesh.element_nodes(field_elem);

            let result = if neighbors.contains(&j) {
                // Use high-order fixed quadrature for neighbors
                regular_integration_fixed_order(
                    source_point,
                    source_normal,
                    &element_coords,
                    field_elem.element_type,
                    &physics,
                    12, // High order
                )
            } else {
                regular_integration(
                    source_point,
                    source_normal,
                    &element_coords,
                    field_elem.element_type,
                    field_elem.area,
                    &physics,
                    None,
                    0,
                    false,
                )
            };

            offdiag_e_high += result.d2g_dnxdny_integral;
            if neighbors.contains(&j) {
                neighbor_e_high += result.d2g_dnxdny_integral;
            }
        }

        let e1_high = self_e + offdiag_e_high;

        println!("\n  High-order neighbor integration (order=12):");
        println!("    Neighbor E sum: {:.4}", neighbor_e_high.re);
        println!("    E[1] closure diagnostic = {:.4}", e1_high.re);

        // Compare neighbor contributions
        let neighbor_diff = (neighbor_e_high - neighbor_e_standard).norm();
        println!(
            "\n  Neighbor E difference (high vs standard): {:.4}",
            neighbor_diff
        );
        let neighbor_error = relative_error(&[neighbor_e_standard], &[neighbor_e_high]);
        println!("  Adaptive/high-order neighbor parity: {neighbor_error:.3e}");
        assert!(
            neighbor_error < 1e-4,
            "adaptive neighbor integration must match order-12 Duffy quadrature within 0.01%; subdivisions={subdivisions}, error={neighbor_error:.3e}"
        );

        println!();
    }

    println!("=== Conclusion ===");
    println!("Adaptive neighbor integration agrees with an independent order-12 Duffy rule.");
}

fn main() {
    let result = std::panic::catch_unwind(|| {
        test_neighbor_integration();
    });
    match result {
        Ok(()) => {
            println!("SUCCESS");
        }
        Err(_) => {
            println!("FAILED");
            std::process::exit(1);
        }
    }
}
