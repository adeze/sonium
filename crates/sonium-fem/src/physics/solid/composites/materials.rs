//! Orthotropic ply and laminate stiffness models.

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OrthotropicPly {
    pub e1: f64,
    pub e2: f64,
    pub g12: f64,
    pub nu12: f64,
    pub thickness: f64,
    pub angle_rad: f64,
}

impl OrthotropicPly {
    pub fn reduced_stiffness(&self) -> [[f64; 3]; 3] {
        let nu21 = self.nu12 * self.e2 / self.e1;
        let denom = 1.0 - self.nu12 * nu21;
        [
            [self.e1 / denom, self.nu12 * self.e2 / denom, 0.0],
            [self.nu12 * self.e2 / denom, self.e2 / denom, 0.0],
            [0.0, 0.0, self.g12],
        ]
    }

    pub fn transformed_stiffness(&self) -> [[f64; 3]; 3] {
        // Classical lamination transformed Q-bar for plane stress.
        let q = self.reduced_stiffness();
        let q11 = q[0][0];
        let q22 = q[1][1];
        let q12 = q[0][1];
        let q66 = q[2][2];
        let m = self.angle_rad.cos();
        let n = self.angle_rad.sin();
        let m2 = m * m;
        let n2 = n * n;
        let m4 = m2 * m2;
        let n4 = n2 * n2;
        let mn = m * n;
        [
            [
                q11 * m4 + 2.0 * (q12 + 2.0 * q66) * m2 * n2 + q22 * n4,
                (q11 + q22 - 4.0 * q66) * m2 * n2 + q12 * (m4 + n4),
                (q11 - q12 - 2.0 * q66) * m2 * mn - (q22 - q12 - 2.0 * q66) * n2 * mn,
            ],
            [
                (q11 + q22 - 4.0 * q66) * m2 * n2 + q12 * (m4 + n4),
                q11 * n4 + 2.0 * (q12 + 2.0 * q66) * m2 * n2 + q22 * m4,
                (q11 - q12 - 2.0 * q66) * n2 * mn - (q22 - q12 - 2.0 * q66) * m2 * mn,
            ],
            [
                (q11 - q12 - 2.0 * q66) * m2 * mn - (q22 - q12 - 2.0 * q66) * n2 * mn,
                (q11 - q12 - 2.0 * q66) * n2 * mn - (q22 - q12 - 2.0 * q66) * m2 * mn,
                (q11 + q22 - 2.0 * q12 - 2.0 * q66) * m2 * n2 + q66 * (m4 + n4),
            ],
        ]
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Laminate {
    pub plies: Vec<OrthotropicPly>,
}

impl Laminate {
    pub fn total_thickness(&self) -> f64 {
        self.plies.iter().map(|ply| ply.thickness).sum()
    }

    pub fn membrane_stiffness(&self) -> [[f64; 3]; 3] {
        self.stiffness_abd().a
    }

    pub fn coupling_stiffness(&self) -> [[f64; 3]; 3] {
        self.stiffness_abd().b
    }

    pub fn bending_stiffness(&self) -> [[f64; 3]; 3] {
        self.stiffness_abd().d
    }

    pub fn stiffness_abd(&self) -> LaminateStiffness {
        let mut a = [[0.0; 3]; 3];
        let mut b = [[0.0; 3]; 3];
        let mut d = [[0.0; 3]; 3];
        let total_thickness = self.total_thickness();
        let mut z0 = -0.5 * total_thickness;
        for ply in &self.plies {
            let z1 = z0 + ply.thickness;
            let q = ply.transformed_stiffness();
            for i in 0..3 {
                for j in 0..3 {
                    a[i][j] += q[i][j] * ply.thickness;
                    b[i][j] += 0.5 * q[i][j] * (z1 * z1 - z0 * z0);
                    d[i][j] += q[i][j] * (z1 * z1 * z1 - z0 * z0 * z0) / 3.0;
                }
            }
            z0 = z1;
        }
        LaminateStiffness { a, b, d }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LaminateStiffness {
    pub a: [[f64; 3]; 3],
    pub b: [[f64; 3]; 3],
    pub d: [[f64; 3]; 3],
}

/// Orthotropic ply strength allowables for first-ply failure indices.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OrthotropicStrength {
    pub x_tension: f64,
    pub x_compression: f64,
    pub y_tension: f64,
    pub y_compression: f64,
    pub shear: f64,
}

impl OrthotropicStrength {
    pub fn tsai_hill_index(&self, stress: [f64; 3]) -> f64 {
        let x = if stress[0] >= 0.0 {
            self.x_tension
        } else {
            self.x_compression
        };
        let y = if stress[1] >= 0.0 {
            self.y_tension
        } else {
            self.y_compression
        };
        stress[0] * stress[0] / (x * x) - stress[0] * stress[1] / (x * x)
            + stress[1] * stress[1] / (y * y)
            + stress[2] * stress[2] / (self.shear * self.shear)
    }

    pub fn max_stress_index(&self, stress: [f64; 3]) -> f64 {
        let sx = if stress[0] >= 0.0 {
            stress[0] / self.x_tension
        } else {
            -stress[0] / self.x_compression
        };
        let sy = if stress[1] >= 0.0 {
            stress[1] / self.y_tension
        } else {
            -stress[1] / self.y_compression
        };
        let sxy = stress[2].abs() / self.shear;
        sx.max(sy).max(sxy)
    }
}
