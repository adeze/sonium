//! Diagnose the original Quad4 transmission system without relaxing its gate.

// Rust guideline compliant 2026-02-21

use sonium_bem::core::assembly::build_transmission_tbem_system_with_continuous_quadratic;
use sonium_bem::core::incident::IncidentField;
use sonium_bem::core::mesh::generate_cubed_sphere_quad_mesh;
use sonium_bem::core::types::PhysicsParams;
use sonium_solvers::direct::lu_solve;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut failed = false;
    for resolution in [2, 4, 8] {
        let mesh = generate_cubed_sphere_quad_mesh(0.2, resolution);
        let system = build_transmission_tbem_system_with_continuous_quadratic(
            &mesh.elements,
            &mesh.nodes,
            PhysicsParams::new(420.0, 343.0, 1.21, false),
            PhysicsParams::new(420.0, 1480.0, 1000.0, true),
        )?;
        let rhs = system.incident_rhs(&IncidentField::plane_wave_z())?;
        let matrix = system.matrix();
        let dimension = system.total_dofs();
        let values = lu_solve(matrix, &rhs)?;
        let mut error_squared = 0.0;
        for row in 0..dimension {
            let action = (0..dimension)
                .map(|col| matrix[[row, col]] * values[col])
                .sum::<num_complex::Complex64>();
            error_squared += (action - rhs[row]).norm_sqr();
        }
        let residual = (error_squared / rhs.iter().map(|v| v.norm_sqr()).sum::<f64>()).sqrt();
        let pressure_max = values[..system.interface_dofs()]
            .iter()
            .map(|v| v.norm())
            .fold(0.0, f64::max);
        let velocity_max = values[system.interface_dofs()..]
            .iter()
            .map(|v| v.norm())
            .fold(0.0, f64::max);
        println!(
            "resolution={resolution}, dimension={dimension}, true_residual={residual:.12e}, max_pressure_trace={pressure_max:.12e}, max_velocity_trace={velocity_max:.12e}"
        );
        failed |= !residual.is_finite() || residual > 1e-8;
    }
    if failed {
        return Err("original 1e-8 transmission equation budget is not met".into());
    }
    Ok(())
}
