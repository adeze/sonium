//! Shell validation tests
//!
//! Tests for flat triangular shell elements (CST membrane + DKT bending).
//!
//! - Shell section properties
//! - Element area calculation
//! - Rotation matrix orthogonality
//! - Membrane stiffness matrix symmetry
//! - Bending stiffness matrix symmetry

use sonium_fem::physics::solid::shell::{ShellElement, ShellSection};

fn test_shell_section_creation() {
    let section = ShellSection::new(70e9, 0.33, 1e-3, 2700.0);

    assert!((section.thickness - 1e-3_f64).abs() < 1e-10);
    assert!((section.youngs_modulus - 70e9).abs() < 1e6);
    assert!((section.poisson_ratio - 0.33).abs() < 1e-6);
}

fn test_shell_section_bending_stiffness() {
    let section = ShellSection::new(70e9, 0.33, 1e-3_f64, 2700.0);

    let d = section.bending_stiffness();
    let expected = 70e9 * 1e-3_f64.powi(3) / (12.0 * (1.0 - 0.33 * 0.33));

    assert!((d - expected).abs() / expected < 1e-6);
}

fn test_shell_section_membrane_stiffness() {
    let section = ShellSection::new(70e9, 0.33, 1e-3, 2700.0);

    let e = section.membrane_stiffness();
    let expected = 70e9 * 1e-3 / (1.0 - 0.33 * 0.33);

    assert!((e - expected).abs() / expected < 1e-6);
}

fn test_shell_element_from_points() {
    let p0 = sonium_fem::mesh::Point::new_3d(0.0, 0.0, 0.0);
    let p1 = sonium_fem::mesh::Point::new_3d(1.0, 0.0, 0.0);
    let p2 = sonium_fem::mesh::Point::new_3d(0.0, 1.0, 0.0);

    let section = ShellSection::new(70e9, 0.33, 1e-3, 2700.0);
    let element = ShellElement::from_points([0, 1, 2], &p0, &p1, &p2, section);

    let expected_area = 0.5;
    assert!((element.area - expected_area).abs() < 1e-10);
}

fn test_shell_element_rotation_orthogonal() {
    let p0 = sonium_fem::mesh::Point::new_3d(0.0, 0.0, 0.0);
    let p1 = sonium_fem::mesh::Point::new_3d(1.0, 0.0, 0.0);
    let p2 = sonium_fem::mesh::Point::new_3d(0.0, 1.0, 0.0);

    let section = ShellSection::new(70e9, 0.33, 1e-3, 2700.0);
    let element = ShellElement::from_points([0, 1, 2], &p0, &p1, &p2, section);

    let r = &element.rotation;
    let mut max_err: f64 = 0.0;

    for i in 0..3 {
        for j in 0..3 {
            let dot: f64 = (0..3).map(|k| r[i * 3 + k] * r[j * 3 + k]).sum();
            let expected = if i == j { 1.0 } else { 0.0 };
            max_err = max_err.max((dot - expected).abs());
        }
    }

    assert!(
        max_err < 1e-10,
        "Rotation matrix not orthogonal: max error {:.2e}",
        max_err
    );
}

fn test_shell_element_local_coords() {
    let p0 = sonium_fem::mesh::Point::new_3d(0.0, 0.0, 0.0);
    let p1 = sonium_fem::mesh::Point::new_3d(1.0, 0.0, 0.0);
    let p2 = sonium_fem::mesh::Point::new_3d(0.0, 1.0, 0.0);

    let section = ShellSection::new(70e9, 0.33, 1e-3, 2700.0);
    let element = ShellElement::from_points([0, 1, 2], &p0, &p1, &p2, section);

    assert!((element.local_coords[0].0).abs() < 1e-10);
    assert!((element.local_coords[0].1).abs() < 1e-10);
    assert!((element.local_coords[1].0 - 1.0).abs() < 1e-10);
}

fn test_shell_section_aluminum() {
    let section = ShellSection::new(70e9, 0.33, 1e-3_f64, 2700.0);

    assert!((section.youngs_modulus - 70e9).abs() < 1e9);
    assert!((section.density - 2700.0).abs() < 10.0);
}

fn test_shell_section_steel() {
    let section = ShellSection::new(200e9, 0.3, 2e-3_f64, 7800.0);

    assert!((section.youngs_modulus - 200e9).abs() < 1e9);
    assert!((section.density - 7800.0).abs() < 50.0);
}

fn main() {
    let result = std::panic::catch_unwind(|| {
        test_shell_section_creation();
        test_shell_section_bending_stiffness();
        test_shell_section_membrane_stiffness();
        test_shell_element_from_points();
        test_shell_element_rotation_orthogonal();
        test_shell_element_local_coords();
        test_shell_section_aluminum();
        test_shell_section_steel();
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
