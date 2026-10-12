//! Development probe for the full original CRASH box native dynamics.
//! Prints native histories for review; does not declare an accuracy gate.
// Rust guideline compliant 2026-02-21
use serde_json::Value;
use sonium_fem::mesh::Point;
use sonium_fem::physics::solid::plasticity::law_metals::{
    Law2, Law2Flow, Law2State, MaxRateEffect,
};
use sonium_fem::physics::solid::plasticity::update::{StressUpdate, StressUpdateModel};
use sonium_fem::physics::solid::shell::crash::{
    DefShell, ElasticShearHook, HourglassParams, LayerStates, Q4CrashGeom, ShellResultants,
    q4_crash_forces,
};
use sonium_fem::physics::solid::shell::{QuadShellElement, ShellSection};
use std::path::Path;

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    (0..3).map(|i| a[i] * b[i]).sum()
}
fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}
fn add(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    std::array::from_fn(|i| a[i] + b[i])
}
fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    std::array::from_fn(|i| a[i] - b[i])
}
fn mv(a: &[[f64; 3]; 3], b: [f64; 3]) -> [f64; 3] {
    std::array::from_fn(|i| dot(a[i], b))
}
fn transpose(a: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
    std::array::from_fn(|i| std::array::from_fn(|j| a[j][i]))
}
fn inverse(a: &[[f64; 3]; 3], b: [f64; 3]) -> Result<[f64; 3], String> {
    let c0 = cross(a[1], a[2]);
    let c1 = cross(a[2], a[0]);
    let c2 = cross(a[0], a[1]);
    let determinant = dot(a[0], c0);
    if !determinant.is_finite() || determinant <= 0. {
        return Err("invalid body inertia".into());
    }
    Ok(std::array::from_fn(|i| {
        (c0[i] * b[0] + c1[i] * b[1] + c2[i] * b[2]) / determinant
    }))
}
fn rotate(a: &mut [[f64; 3]; 3], omega: [f64; 3], dt: f64) {
    let norm = dot(omega, omega).sqrt();
    if norm == 0. {
        return;
    }
    let axis = omega.map(|v| v / norm);
    let angle = norm * dt;
    for column in 0..3 {
        let old = [a[0][column], a[1][column], a[2][column]];
        let rotated = add(
            add(
                old.map(|v| v * angle.cos()),
                cross(axis, old).map(|v| v * angle.sin()),
            ),
            axis.map(|v| v * dot(axis, old) * (1. - angle.cos())),
        );
        for i in 0..3 {
            a[i][column] = rotated[i];
        }
    }
}
fn local(geom: &Q4CrashGeom, v: [f64; 3]) -> [f64; 3] {
    [dot(geom.ex, v), dot(geom.ey, v), dot(geom.ez, v)]
}
fn global(geom: &Q4CrashGeom, v: [f64; 3]) -> [f64; 3] {
    std::array::from_fn(|i| geom.ex[i] * v[0] + geom.ey[i] * v[1] + geom.ez[i] * v[2])
}
fn rotate_inplane(s: &mut [f64; 6], angle: f64) {
    let c = angle.cos();
    let sn = angle.sin();
    let old = *s;
    s[0] = c * c * old[0] + sn * sn * old[1] - 2. * c * sn * old[3];
    s[1] = sn * sn * old[0] + c * c * old[1] + 2. * c * sn * old[3];
    s[3] = c * sn * (old[0] - old[1]) + (c * c - sn * sn) * old[3];
    s[4] = sn * old[5] + c * old[4];
    s[5] = c * old[5] - sn * old[4];
}
fn rotate_resultant(s: &mut [f64; 3], angle: f64) {
    let mut tensor = [s[0], s[1], 0., s[2], 0., 0.];
    rotate_inplane(&mut tensor, angle);
    *s = [tensor[0], tensor[1], tensor[3]];
}
// Symmetric convected frame bisects the two current covariant directions.
// Geometry is still evaluated by the production Q4 center operator.
fn symmetric_geometry(
    corners: [[f64; 3]; 4],
    section: ShellSection,
) -> Result<Q4CrashGeom, String> {
    let points = corners.map(|p| Point::new_3d(p[0], p[1], p[2]));
    let mut quad = QuadShellElement::from_points_with_section(
        [0, 1, 2, 3],
        &points[0],
        &points[1],
        &points[2],
        &points[3],
        section,
    );
    let a = sub(add(corners[1], corners[2]), add(corners[0], corners[3]));
    let b = sub(add(corners[2], corners[3]), add(corners[0], corners[1]));
    let normalize = |v: [f64; 3]| -> Result<[f64; 3], String> {
        let n = dot(v, v).sqrt();
        if !n.is_finite() || n <= 0. {
            return Err("degenerate convected frame".into());
        }
        Ok(v.map(|x| x / n))
    };
    let ez = normalize(cross(a, b))?;
    let ex = normalize(add(normalize(a)?, cross(normalize(b)?, ez)))?;
    let ey = cross(ez, ex);
    quad.rotation = [
        ex[0], ey[0], ez[0], ex[1], ey[1], ez[1], ex[2], ey[2], ez[2],
    ];
    for (n, p) in corners.iter().enumerate() {
        let d = sub(*p, quad.origin);
        quad.local_coords[n] = (dot(d, ex), dot(d, ey));
        quad.warp[n] = dot(d, ez);
    }
    Q4CrashGeom::from_quad(&quad).map_err(|e| e.to_string())
}
// Experimental deck-selected plane-stress projection. Kept in the probe
// until full response and independent material checks justify a public API.
struct BoxMaterial {
    law: Law2,
    projection: bool,
    shell_rate: Option<f64>,
}
impl std::ops::Deref for BoxMaterial {
    type Target = Law2;
    fn deref(&self) -> &Law2 {
        &self.law
    }
}
impl StressUpdateModel for BoxMaterial {
    type State = Law2State;
    type StrainIncrement = [f64; 6];
    fn update(&self, state: &Law2State, d: &[f64; 6]) -> StressUpdate {
        self.update_with_rate(state, d, 1.)
    }
    fn update_with_rate(&self, state: &Law2State, d: &[f64; 6], dt: f64) -> StressUpdate {
        if !self.projection {
            return self.law.update_with_rate(state, d, dt);
        }
        let e = self.youngs_modulus;
        let nu = self.poisson_ratio;
        let a = e / (1. - nu * nu);
        let b = nu * a;
        let g = self.shear_modulus();
        let mut stress = state.stress;
        stress[0] += a * d[0] + b * d[1];
        stress[1] += b * d[0] + a * d[1];
        stress[2] = 0.;
        stress[3] += g * d[3];
        let q = (stress[0].powi(2) + stress[1].powi(2) - stress[0] * stress[1]
            + 3. * stress[3].powi(2))
        .sqrt();
        // The deck's unfiltered total-rate option uses the largest
        // in-plane tensor increment; transverse shear stays elastic.
        let rate = d[0].abs().max(d[1].abs()).max(0.5 * d[3].abs()) / dt;
        let rate = self.shell_rate.unwrap_or(rate);
        let yield_stress = self.flow_stress(state.equiv_plastic_strain, rate, state.temperature);
        let yielding = q > yield_stress;
        let dep = if yielding { (q - yield_stress) / e } else { 0. };
        if yielding {
            let factor = yield_stress / q;
            for i in [0, 1, 3] {
                stress[i] *= factor;
            }
        }
        // This explicit projection has no implicit Newton tangent.
        // The elastic plane-stress tangent is sufficient for the driver,
        // which sees sigma_zz = 0 at every candidate increment.
        let mut tangent = [0.; 36];
        tangent[0] = a;
        tangent[1] = b;
        tangent[6] = b;
        tangent[7] = a;
        tangent[14] = e;
        tangent[21] = g;
        StressUpdate {
            stress,
            tangent,
            delta_equiv_plastic_strain: dep,
            yielding,
        }
    }
    fn is_rate_dependent(&self) -> bool {
        true
    }
    fn commit(&self, state: &mut Law2State, update: &StressUpdate) {
        self.law.commit(state, update);
    }
}
struct BoxModel {
    reference: Vec<[f64; 3]>,
    points: Vec<[f64; 3]>,
    velocity: Vec<[f64; 6]>,
    mass: Vec<[f64; 6]>,
    quads: Vec<[usize; 4]>,
    layers: Vec<LayerStates<Law2State>>,
    previous: Vec<ShellResultants>,
    axes: Vec<[[f64; 3]; 3]>,
    hourglass_history: Vec<[f64; 3]>,
    body: Vec<usize>,
    wall: Vec<usize>,
    rotation: [[f64; 3]; 3],
    inertia: [[f64; 3]; 3],
    body_mass: f64,
    material: BoxMaterial,
    density: f64,
    thickness: f64,
    internal_work: f64,
    plastic_work: f64,
    hourglass_work: f64,
    impulse: f64,
}
impl BoxModel {
    fn new(meta: &Value) -> Result<Self, String> {
        let scalar = |key: &str| {
            meta["law2"][key]
                .as_f64()
                .ok_or_else(|| format!("missing {key}"))
        };
        let points = meta["nodes"]
            .as_array()
            .ok_or("missing nodes")?
            .iter()
            .map(|r| {
                Ok([
                    r[1].as_f64().ok_or("x")?,
                    r[2].as_f64().ok_or("y")?,
                    r[3].as_f64().ok_or("z")?,
                ])
            })
            .collect::<Result<Vec<_>, String>>()?;
        let quads = meta["quads"]
            .as_array()
            .ok_or("missing quads")?
            .iter()
            .map(|r| std::array::from_fn(|i| r[i + 1].as_u64().unwrap() as usize - 1))
            .collect::<Vec<_>>();
        let group = |id: &str| {
            meta["node_groups"][id]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_u64().unwrap() as usize - 1)
                .collect::<Vec<_>>()
        };
        let body = group("4");
        let wall = group("3");
        let thickness = meta["thickness_m"].as_f64().ok_or("thickness")?;
        let density = scalar("density_kg_m3")?;
        let mut mass = vec![[0.; 6]; points.len()];
        for q in &quads {
            let geom = Q4CrashGeom::from_corners(q.map(|node| points[node]), thickness)
                .map_err(|e| e.to_string())?;
            let mut lumped = geom.lumped_masses(density);
            if std::env::var("CRASH_BOX_ROTARY").as_deref() == Ok("1") {
                // The original one-point explicit shell adds numerical
                // rotational inertia m_node * area / 9 for stability.
                for node in 0..4 {
                    for d in 3..6 {
                        lumped[6 * node + d] += lumped[6 * node] * geom.area / 9.;
                    }
                }
            }
            for (slot, node) in q.iter().enumerate() {
                for d in 0..6 {
                    mass[*node][d] += lumped[6 * slot + d];
                }
            }
        }
        for node in group("6") {
            for d in 0..3 {
                mass[node][d] += meta["added_mass_kg"].as_f64().unwrap();
            }
        }
        let master = 154;
        for d in 0..3 {
            mass[master][d] += meta["rigid_body_card_original"][0][4].as_f64().unwrap();
        }
        let mut inertia = [[0.; 3]; 3];
        let mut body_mass = mass[master][0];
        let mut first_moment = [0.; 3];
        for node in &body {
            let r = sub(points[*node], points[master]);
            let m = mass[*node][0];
            body_mass += m;
            first_moment = add(first_moment, r.map(|v| m * v));
            for i in 0..3 {
                for j in 0..3 {
                    inertia[i][j] += if i == j {
                        m * dot(r, r) + mass[*node][3 + i]
                    } else {
                        0.
                    } - m * r[i] * r[j];
                }
            }
        }
        if dot(first_moment, first_moment).sqrt() > 1e-12 {
            return Err("body master is not the mass center".into());
        }
        let material = Law2::new(
            scalar("young_pa")?,
            scalar("poisson")?,
            Law2Flow::JohnsonCook {
                a: scalar("a_pa")?,
                b: scalar("b_pa")?,
                n: scalar("n")?,
                c: scalar("c")?,
            },
            scalar("reference_rate_s")?,
            0.,
            0.,
            298.,
            0.,
            scalar("max_stress_pa")?,
            MaxRateEffect::Scaled,
            1e30,
            0.,
            1e30,
        );
        let layers = quads
            .iter()
            .map(|_| {
                if std::env::var("CRASH_BOX_LAYERS").as_deref() == Ok("1") {
                    LayerStates::new_piecewise_linear(Law2State::new(298.), 3, thickness)
                } else {
                    LayerStates::new(Law2State::new(298.), 3, thickness)
                }
                .map_err(|e| e.to_string())
            })
            .collect::<Result<Vec<_>, _>>()?;
        let initial = meta["initial_velocity_m_s"][0].as_f64().unwrap();
        let axes = quads
            .iter()
            .map(|q| {
                let g = Q4CrashGeom::from_corners(q.map(|node| points[node]), thickness).unwrap();
                [g.ex, g.ey, g.ez]
            })
            .collect();
        let mut model = Self {
            reference: points.clone(),
            points,
            velocity: vec![[initial, 0., 0., 0., 0., 0.]; mass.len()],
            mass,
            previous: vec![ShellResultants::default(); quads.len()],
            axes,
            hourglass_history: vec![[0.; 3]; quads.len()],
            quads,
            layers,
            body,
            wall,
            rotation: [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]],
            inertia,
            body_mass,
            material: BoxMaterial {
                law: material,
                shell_rate: None,
                projection: std::env::var("CRASH_BOX_PROJECTION").as_deref() == Ok("1"),
            },
            density,
            thickness,
            internal_work: 0.,
            plastic_work: 0.,
            hourglass_work: 0.,
            impulse: 0.,
        };
        model.project_wall();
        Ok(model)
    }
    fn project_wall(&mut self) {
        for node in &self.wall {
            if self.points[*node][0] <= 1e-15 && self.velocity[*node][0] < 0. {
                self.impulse += self.mass[*node][0] * self.velocity[*node][0];
                self.velocity[*node][0] = 0.;
            }
        }
    }
    fn body_kinematics(&mut self) {
        let master = 154;
        let omega = [
            self.velocity[master][3],
            self.velocity[master][4],
            self.velocity[master][5],
        ];
        for node in &self.body {
            let r = mv(
                &self.rotation,
                sub(self.reference[*node], self.reference[master]),
            );
            self.points[*node] = add(self.points[master], r);
            let spin = cross(omega, r);
            for i in 0..3 {
                self.velocity[*node][i] = self.velocity[master][i] + spin[i];
                self.velocity[*node][3 + i] = omega[i];
            }
        }
    }
    fn acceleration(&self, force: &[[f64; 6]]) -> Result<Vec<[f64; 6]>, String> {
        let mut out = vec![[0.; 6]; self.points.len()];
        for node in 0..154 {
            if self.body.contains(&node) {
                continue;
            }
            for i in 0..6 {
                out[node][i] = force[node][i] / self.mass[node][i];
            }
        }
        let master = 154;
        let mut f = [0.; 3];
        let mut torque = [0.; 3];
        for node in &self.body {
            let linear = [force[*node][0], force[*node][1], force[*node][2]];
            let moment = [force[*node][3], force[*node][4], force[*node][5]];
            f = add(f, linear);
            torque = add(
                torque,
                add(
                    moment,
                    cross(sub(self.points[*node], self.points[master]), linear),
                ),
            );
        }
        let omega = [
            self.velocity[master][3],
            self.velocity[master][4],
            self.velocity[master][5],
        ];
        let r_t = transpose(&self.rotation);
        let angular_momentum = mv(&self.rotation, mv(&self.inertia, mv(&r_t, omega)));
        let local_torque = mv(&r_t, sub(torque, cross(omega, angular_momentum)));
        let angular = mv(&self.rotation, inverse(&self.inertia, local_torque)?);
        for i in 0..3 {
            out[master][i] = f[i] / self.body_mass;
            out[master][i + 3] = angular[i];
        }
        Ok(out)
    }
    fn force(&mut self, dt: f64) -> Result<Vec<[f64; 6]>, String> {
        let mut force = vec![[0.; 6]; self.points.len()];
        let def = DefShell::explicit_defaults();
        let shear = ElasticShearHook::new(self.material.shear_modulus());
        // LAW2's shell stabilization uses the uniaxial wave speed (MAT02 SSP),
        // rather than the plane-stress dilatational speed.
        let sound = (self.material.youngs_modulus / self.density).sqrt();
        for (idx, q) in self.quads.iter().enumerate() {
            let convected = std::env::var("CRASH_BOX_FRAME").as_deref() == Ok("1");
            let geom = if convected {
                symmetric_geometry(
                    q.map(|node| self.points[node]),
                    ShellSection::new(
                        self.material.youngs_modulus,
                        self.material.poisson_ratio,
                        self.thickness,
                        self.density,
                    ),
                )?
            } else {
                Q4CrashGeom::from_corners(q.map(|node| self.points[node]), self.thickness)
                    .map_err(|e| e.to_string())?
            };
            let mut u = [0.; 24];
            let mut v = [0.; 24];
            for (slot, node) in q.iter().enumerate() {
                let lv = local(
                    &geom,
                    [
                        self.velocity[*node][0],
                        self.velocity[*node][1],
                        self.velocity[*node][2],
                    ],
                );
                let lw = local(
                    &geom,
                    [
                        self.velocity[*node][3],
                        self.velocity[*node][4],
                        self.velocity[*node][5],
                    ],
                );
                let lu = local(&geom, sub(self.points[*node], self.reference[*node]));
                for i in 0..3 {
                    v[6 * slot + i] = lv[i];
                    v[6 * slot + 3 + i] = lw[i];
                    u[6 * slot + i] = lu[i];
                }
            }
            // Jaumann spin relative to the convecting edge-aligned frame.
            // A rigid rotation has equal material/frame spins, hence zero
            // relative correction; a shear deformation generally does not.
            let mut spin = 0.;
            for node in 0..4 {
                spin += 0.5
                    * (geom.membrane_b[16 + 2 * node + 1] * v[6 * node + 1]
                        - geom.membrane_b[16 + 2 * node] * v[6 * node]);
            }
            let old_axes = self.axes[idx];
            let frame_angle = dot(geom.ex, old_axes[1]).atan2(dot(geom.ex, old_axes[0]));
            // The reference symmetric convected formulation stores stresses
            // in its evolving material frame. This option isolates that
            // transport from the default Jaumann correction.
            let relative_spin = if convected {
                0.
            } else {
                spin * dt - frame_angle
            };
            for state in &mut self.layers[idx].states {
                rotate_inplane(&mut state.stress, relative_spin);
                rotate_inplane(&mut state.back_stress, relative_spin);
            }
            let [gx, gy] = self.layers[idx].transverse_shear;
            self.layers[idx].transverse_shear = [
                relative_spin.cos() * gx - relative_spin.sin() * gy,
                relative_spin.sin() * gx + relative_spin.cos() * gy,
            ];
            rotate_resultant(&mut self.previous[idx].membrane, relative_spin);
            rotate_resultant(&mut self.previous[idx].bending, relative_spin);
            let [qx, qy] = self.previous[idx].shear;
            self.previous[idx].shear = [
                relative_spin.cos() * qx - relative_spin.sin() * qy,
                relative_spin.sin() * qx + relative_spin.cos() * qy,
            ];
            self.axes[idx] = [geom.ex, geom.ey, geom.ez];
            let mut hg = HourglassParams::for_shell(
                def.formulation,
                self.density,
                sound,
                self.material.shear_modulus(),
                geom.area,
                self.thickness,
                geom.char_length,
                false,
            )
            .map_err(|e| e.to_string())?;
            if std::env::var("CRASH_BOX_HG").as_deref() == Ok("1") {
                // Original blank Hm/Hf resolve to 0.01, versus 0.05
                // in the native convenience constructor.
                hg.membrane_viscous *= 0.2;
                hg.bending_viscous *= 0.2;
            }
            let card_hg = std::env::var("CRASH_BOX_HG").as_deref() == Ok("2");
            // A matched elastic counterfactual can isolate all stabilization
            // from physical shell forces without changing the production API.
            let no_hg = std::env::var("CRASH_BOX_HG").as_deref() == Ok("3");
            if card_hg || no_hg {
                hg = HourglassParams {
                    membrane_viscous: 0.,
                    membrane_stiffness: 0.,
                    bending_viscous: 0.,
                    bending_stiffness: 0.,
                    drilling: 0.,
                    ..HourglassParams::default()
                };
            }
            if std::env::var("CRASH_BOX_RATE").as_deref() == Ok("1") {
                // Energy-equivalent section rate, shared by all thickness
                // points: integrate the squared incompressible membrane
                // strain rate plus the squared linear bending strain rate.
                let mut em = [0.; 3];
                let mut k = [0.; 3];
                for n in 0..4 {
                    for row in 0..3 {
                        em[row] += geom.membrane_b[row * 8 + 2 * n] * v[6 * n]
                            + geom.membrane_b[row * 8 + 2 * n + 1] * v[6 * n + 1];
                        for d in 0..3 {
                            k[row] += geom.bending_b[row * 12 + 3 * n + d] * v[6 * n + 2 + d];
                        }
                    }
                }
                let norm =
                    |r: [f64; 3]| r[0] * r[0] + r[1] * r[1] + r[0] * r[1] + 0.25 * r[2] * r[2];
                self.material.shell_rate =
                    Some((4. * norm(em) / 3. + self.thickness.powi(2) * norm(k) / 9.).sqrt());
            }
            let mut out = q4_crash_forces(
                &geom,
                &mut self.layers[idx],
                &self.material,
                &def,
                &u,
                &v,
                dt,
                &hg,
                &shear,
            )
            .map_err(|e| e.to_string())?;
            if card_hg {
                // Independent modal stabilization equations for the original
                // Hm=Hf=Hr=0.01 explicit viscoelastic shell card.
                let h = 0.01;
                let rho = self.density;
                let t = self.thickness;
                let length = geom.area.sqrt();
                let factor = (5. / (18. * (1. + self.material.poisson_ratio))).sqrt();
                let c_mem = 0.;
                let q_mem = rho * 100. * 0.5 * h * t * length / 4.;
                let c_w = 0.;
                let q_w = rho * 100. * 0.5 * h * t * t * factor / 4.;
                let c_rot = rho * sound * h.sqrt() * t * t * geom.area * (3_f64.sqrt() / 24.)
                    / (4. * 2_f64.sqrt());
                let k_mem = 0.5 * self.material.youngs_modulus * h * t / 4.;
                let k_w = 0.5 * self.material.youngs_modulus * h * factor * factor * t.powi(3)
                    / (4. * geom.area);
                let gamma = geom.hg_vector(def.formulation);
                let [hx, hy, hz] = self.hourglass_history[idx];
                self.hourglass_history[idx] = [
                    relative_spin.cos() * hx - relative_spin.sin() * hy,
                    relative_spin.sin() * hx + relative_spin.cos() * hy,
                    hz,
                ];
                for d in 0..5 {
                    let modal = (0..4).map(|n| gamma[n] * v[6 * n + d]).sum::<f64>();
                    let c = if d < 2 {
                        c_mem
                    } else if d == 2 {
                        c_w
                    } else {
                        c_rot
                    };
                    let quadratic = if d < 2 {
                        q_mem
                    } else if d == 2 {
                        q_w
                    } else {
                        rho * 0.1 * 0.5 * h * t * t * geom.area * (3_f64.sqrt() / 24.) / 4.
                    };
                    let mut modal_force = (c + quadratic * modal.abs()) * modal;
                    if d < 3 {
                        let k = if d < 2 { k_mem } else { k_w };
                        self.hourglass_history[idx][d] += k * modal * dt;
                        modal_force += self.hourglass_history[idx][d];
                    }
                    for n in 0..4 {
                        out.forces[6 * n + d] += gamma[n] * modal_force;
                    }
                    out.hg_power += modal_force * modal;
                }
            }
            let mut mem = [0.; 3];
            let mut curv = [0.; 3];
            let mut transverse = [0.; 2];
            for node in 0..4 {
                for row in 0..3 {
                    mem[row] += geom.membrane_b[row * 8 + 2 * node] * v[6 * node]
                        + geom.membrane_b[row * 8 + 2 * node + 1] * v[6 * node + 1];
                    for j in 0..3 {
                        curv[row] += geom.bending_b[row * 12 + 3 * node + j] * v[6 * node + 2 + j];
                    }
                }
                for (row, total) in transverse.iter_mut().enumerate() {
                    for j in 0..3 {
                        *total += geom.shear_b[row * 12 + 3 * node + j] * v[6 * node + 2 + j];
                    }
                }
            }
            let old = self.previous[idx];
            self.internal_work += 0.5
                * geom.area
                * dt
                * ((0..3)
                    .map(|i| {
                        (old.membrane[i] + out.resultants.membrane[i]) * mem[i]
                            + (old.bending[i] + out.resultants.bending[i]) * curv[i]
                    })
                    .sum::<f64>()
                    + (0..2)
                        .map(|i| (old.shear[i] + out.resultants.shear[i]) * transverse[i])
                        .sum::<f64>());
            self.previous[idx] = out.resultants;
            self.plastic_work += out.plastic_work_rate * dt;
            self.hourglass_work += out.hg_power * dt;
            for (slot, node) in q.iter().enumerate() {
                let f = global(
                    &geom,
                    [
                        out.forces[6 * slot],
                        out.forces[6 * slot + 1],
                        out.forces[6 * slot + 2],
                    ],
                );
                let m = global(
                    &geom,
                    [
                        out.forces[6 * slot + 3],
                        out.forces[6 * slot + 4],
                        out.forces[6 * slot + 5],
                    ],
                );
                for i in 0..3 {
                    force[*node][i] -= f[i];
                    force[*node][i + 3] -= m[i];
                }
            }
        }
        if force.iter().flatten().any(|f| !f.is_finite()) {
            return Err("nonfinite box force".into());
        }
        Ok(force)
    }
    fn kinetic(&self) -> f64 {
        let mut e = 0.;
        for node in 0..154 {
            if self.body.contains(&node) {
                continue;
            }
            for i in 0..6 {
                e += 0.5 * self.mass[node][i] * self.velocity[node][i].powi(2);
            }
        }
        let v = self.velocity[154];
        e += 0.5 * self.body_mass * (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
        let omega = mv(&transpose(&self.rotation), [v[3], v[4], v[5]]);
        e + 0.5 * dot(omega, mv(&self.inertia, omega))
    }
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let mut meta: Value = serde_json::from_str(&std::fs::read_to_string(
        root.join("decks/crash_box_original/model.json"),
    )?)?;
    let golden: Value = serde_json::from_str(&std::fs::read_to_string(
        root.join("goldens/miniqa/CRASH/crashsection/box1_42.json"),
    )?)?;
    if std::env::var("CRASH_BOX_ELASTIC").as_deref() == Ok("1") {
        // Counterfactual keeps the original elasticity, mesh and masses.
        // Match an engine card with a and sigma_max = 1e21 in original
        // stress units (1e30 Pa after the physical similarity transform).
        meta["law2"]["a_pa"] = 1e30.into();
        meta["law2"]["b_pa"] = 0_f64.into();
        meta["law2"]["max_stress_pa"] = 1e30.into();
    }
    let mut model = BoxModel::new(&meta)?;
    let count = std::env::var("CRASH_BOX_STEPS")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(5000);
    if count < 100 {
        return Err("CRASH_BOX_STEPS must be at least 100".into());
    }
    let end = golden["time"]
        .as_array()
        .unwrap()
        .last()
        .unwrap()
        .as_f64()
        .unwrap()
        * 0.001;
    let end = std::env::var("CRASH_BOX_END_S")
        .ok()
        .map(|s| s.parse::<f64>())
        .transpose()?
        .unwrap_or(end);
    if !end.is_finite() || end <= 0. {
        return Err("CRASH_BOX_END_S must be positive finite".into());
    }
    let dt = end / count as f64;
    let mut accel = vec![[0.; 6]; model.points.len()];
    let initial = model.kinetic();
    println!(
        "time_s,internal_j,plastic_j,hourglass_j,kinetic_j,wall_impulse_ns,node17_dx_m,node17_dy_m,node17_dz_m,node83_dx_m,max_plastic"
    );
    let print = |model: &BoxModel, time: f64| {
        let d17 = sub(model.points[16], model.reference[16]);
        let d83 = sub(model.points[82], model.reference[82]);
        let ep = model
            .layers
            .iter()
            .flat_map(|l| l.states.iter())
            .map(|s| s.equiv_plastic_strain)
            .fold(0_f64, f64::max);
        println!(
            "{time:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{ep:.12e}",
            model.internal_work,
            model.plastic_work,
            model.hourglass_work,
            model.kinetic(),
            model.impulse,
            d17[0],
            d17[1],
            d17[2],
            d83[0]
        );
    };
    print(&model, 0.);
    for step in 0..count {
        for (node, a) in accel.iter().enumerate() {
            if model.body.contains(&node) {
                continue;
            }
            for (i, value) in a.iter().enumerate() {
                model.velocity[node][i] += 0.5 * dt * value;
            }
        }
        model.project_wall();
        model.body_kinematics();
        for node in 0..model.points.len() {
            if model.body.contains(&node) {
                continue;
            }
            for i in 0..3 {
                model.points[node][i] += dt * model.velocity[node][i];
            }
        }
        let omega = [
            model.velocity[154][3],
            model.velocity[154][4],
            model.velocity[154][5],
        ];
        rotate(&mut model.rotation, omega, dt);
        model.body_kinematics();
        let force = model.force(dt).map_err(|e| format!("step {step}: {e}"))?;
        accel = model.acceleration(&force)?;
        for (node, a) in accel.iter().enumerate() {
            if model.body.contains(&node) {
                continue;
            }
            for (i, value) in a.iter().enumerate() {
                model.velocity[node][i] += 0.5 * dt * value;
            }
        }
        model.project_wall();
        model.body_kinematics();
        if (step + 1) % ((count / 100).max(1)) == 0 {
            print(&model, (step + 1) as f64 * dt);
        }
    }
    for element in [16, 56] {
        let layer = &model.layers[element - 1];
        eprintln!(
            "element {element}: states={:?}; shear={:?}",
            layer
                .states
                .iter()
                .map(|s| (s.stress, s.equiv_plastic_strain))
                .collect::<Vec<_>>(),
            layer.transverse_shear
        );
    }
    eprintln!(
        "initial KE={initial}; final energy residual={} J",
        model.kinetic() + model.internal_work + model.hourglass_work - initial
    );
    Ok(())
}
