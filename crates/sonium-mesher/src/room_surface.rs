//! Surface meshing for the exchange-room geometry types.
//!
//! The algorithms are intentionally kept out of `sonium-xem`: XEM owns the
//! serializable geometry and mesh data, while this module owns the selectable
//! discretisation policy.

use sonium_xem::{
    LShapedRoom, Point3D, RectangularRoom, RoomGeometry, RoomMesh, Source, SurfaceElement,
};

/// Generate a BEM surface mesh for exchange-room geometries.
///
/// Import this trait to migrate former `sonium_xem` `generate_mesh` and
/// `generate_adaptive_mesh` calls without changing their call syntax.
pub trait RoomSurfaceMeshing {
    /// Generate a uniform quadrilateral surface mesh.
    fn generate_mesh(&self, elements_per_meter: usize) -> RoomMesh;

    /// Generate a wavelength-controlled triangular mesh with source refinement.
    fn generate_adaptive_mesh(
        &self,
        base_elements_per_meter: usize,
        frequency: f64,
        sources: &[Source],
        speed_of_sound: f64,
    ) -> RoomMesh;
}

impl RoomSurfaceMeshing for RoomGeometry {
    fn generate_mesh(&self, elements_per_meter: usize) -> RoomMesh {
        match self {
            Self::Rectangular(room) => room.generate_mesh(elements_per_meter),
            Self::LShaped(room) => room.generate_mesh(elements_per_meter),
        }
    }

    fn generate_adaptive_mesh(
        &self,
        base_elements_per_meter: usize,
        frequency: f64,
        sources: &[Source],
        speed_of_sound: f64,
    ) -> RoomMesh {
        match self {
            Self::Rectangular(room) => room.generate_adaptive_mesh(
                base_elements_per_meter,
                frequency,
                sources,
                speed_of_sound,
            ),
            Self::LShaped(room) => room.generate_adaptive_mesh(
                base_elements_per_meter,
                frequency,
                sources,
                speed_of_sound,
            ),
        }
    }
}

impl RoomSurfaceMeshing for RectangularRoom {
    fn generate_mesh(&self, elements_per_meter: usize) -> RoomMesh {
        let resolution = elements_per_meter.max(1) as f64;
        let surfaces = rectangular_surfaces(self);
        uniform_mesh(&surfaces, resolution)
    }

    fn generate_adaptive_mesh(
        &self,
        base: usize,
        frequency: f64,
        sources: &[Source],
        speed: f64,
    ) -> RoomMesh {
        adaptive_mesh(&rectangular_surfaces(self), base, frequency, sources, speed)
    }
}

impl RoomSurfaceMeshing for LShapedRoom {
    fn generate_mesh(&self, elements_per_meter: usize) -> RoomMesh {
        uniform_mesh(&l_shaped_surfaces(self), elements_per_meter.max(1) as f64)
    }

    fn generate_adaptive_mesh(
        &self,
        base: usize,
        frequency: f64,
        sources: &[Source],
        speed: f64,
    ) -> RoomMesh {
        adaptive_mesh(&l_shaped_surfaces(self), base, frequency, sources, speed)
    }
}

#[derive(Clone, Copy)]
struct Surface {
    origin: Point3D,
    u: Point3D,
    v: Point3D,
}

impl Surface {
    fn lengths(self) -> (f64, f64) {
        (
            self.origin.distance_to(&self.u),
            self.origin.distance_to(&self.v),
        )
    }
}

/// Exact pre-allocation counts for a room surface mesh (no vertices allocated).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RoomSurfaceMeshSize {
    pub nodes: usize,
    pub elements: usize,
}

/// Plan the same grid used by [`RoomSurfaceMeshing`]. `Some(frequency)` selects
/// adaptive triangular meshing, including source-proximity refinement; `None`
/// selects uniform quads. Invalid or unrepresentable sizes return an error.
pub fn room_surface_mesh_size(
    room: &RoomGeometry,
    base: usize,
    frequency: Option<f64>,
    sources: &[Source],
    speed: f64,
) -> Result<RoomSurfaceMeshSize, String> {
    if let Some(frequency) = frequency
        && (!frequency.is_finite() || frequency <= 0.0 || !speed.is_finite() || speed <= 0.0)
    {
        return Err("adaptive mesh frequency and sound speed must be positive and finite".into());
    }
    let surfaces = match room {
        RoomGeometry::Rectangular(room) => rectangular_surfaces(room).to_vec(),
        RoomGeometry::LShaped(room) => l_shaped_surfaces(room).to_vec(),
    };
    let mut size = RoomSurfaceMeshSize {
        nodes: 0,
        elements: 0,
    };
    for surface in surfaces {
        let (nu, nv) = surface_divisions(
            surface,
            base.max(1) as f64,
            frequency.map(|f| speed / f / 8.0),
            sources,
        )?;
        let nodes = nu
            .checked_add(1)
            .and_then(|u| nv.checked_add(1).and_then(|v| u.checked_mul(v)));
        let elements = nu
            .checked_mul(nv)
            .and_then(|n| n.checked_mul(if frequency.is_some() { 2 } else { 1 }));
        size.nodes = nodes
            .and_then(|n| size.nodes.checked_add(n))
            .ok_or("room mesh node count overflow")?;
        size.elements = elements
            .and_then(|n| size.elements.checked_add(n))
            .ok_or("room mesh element count overflow")?;
    }
    Ok(size)
}

fn surface_divisions(
    surface: Surface,
    resolution: f64,
    target: Option<f64>,
    sources: &[Source],
) -> Result<(usize, usize), String> {
    let (u, v) = surface.lengths();
    let refinement = target.is_some_and(|target| {
        sources
            .iter()
            .any(|source| distance_to_surface(&source.position, surface) < target * 2.0)
    });
    let divisions = |length: f64| {
        let base = (length * resolution).ceil();
        let count = target
            .map_or(base, |target| base.max((length / target).ceil()))
            .max(1.0);
        if !length.is_finite() || !count.is_finite() || count >= usize::MAX as f64 {
            return Err("room mesh resolution is unrepresentable".to_string());
        }
        (count as usize)
            .checked_mul(if refinement { 2 } else { 1 })
            .ok_or_else(|| "room mesh refinement overflow".into())
    };
    Ok((divisions(u)?, divisions(v)?))
}

fn rectangular_surfaces(room: &RectangularRoom) -> [Surface; 6] {
    let (w, d, h) = (room.width, room.depth, room.height);
    [
        Surface {
            origin: p(0.0, 0.0, 0.0),
            u: p(0.0, d, 0.0),
            v: p(w, 0.0, 0.0),
        },
        Surface {
            origin: p(0.0, 0.0, h),
            u: p(w, 0.0, h),
            v: p(0.0, d, h),
        },
        Surface {
            origin: p(0.0, 0.0, 0.0),
            u: p(w, 0.0, 0.0),
            v: p(0.0, 0.0, h),
        },
        Surface {
            origin: p(0.0, d, 0.0),
            u: p(0.0, d, h),
            v: p(w, d, 0.0),
        },
        Surface {
            origin: p(0.0, 0.0, 0.0),
            u: p(0.0, 0.0, h),
            v: p(0.0, d, 0.0),
        },
        Surface {
            origin: p(w, 0.0, 0.0),
            u: p(w, d, 0.0),
            v: p(w, 0.0, h),
        },
    ]
}

fn l_shaped_surfaces(room: &LShapedRoom) -> [Surface; 10] {
    let (w1, d1, w2, d2, h) = (
        room.width1,
        room.depth1,
        room.width2,
        room.depth2,
        room.height,
    );
    let depth = d1 + d2;
    [
        Surface {
            origin: p(0.0, 0.0, 0.0),
            u: p(w1, 0.0, 0.0),
            v: p(0.0, d1, 0.0),
        },
        Surface {
            origin: p(0.0, d1, 0.0),
            u: p(w2, d1, 0.0),
            v: p(0.0, depth, 0.0),
        },
        Surface {
            origin: p(0.0, 0.0, h),
            u: p(w1, 0.0, h),
            v: p(0.0, d1, h),
        },
        Surface {
            origin: p(0.0, d1, h),
            u: p(w2, d1, h),
            v: p(0.0, depth, h),
        },
        Surface {
            origin: p(0.0, 0.0, 0.0),
            u: p(w1, 0.0, 0.0),
            v: p(0.0, 0.0, h),
        },
        Surface {
            origin: p(w1, 0.0, 0.0),
            u: p(w1, d1, 0.0),
            v: p(w1, 0.0, h),
        },
        Surface {
            origin: p(0.0, 0.0, 0.0),
            u: p(0.0, depth, 0.0),
            v: p(0.0, 0.0, h),
        },
        Surface {
            origin: p(0.0, depth, 0.0),
            u: p(w2, depth, 0.0),
            v: p(0.0, depth, h),
        },
        Surface {
            origin: p(w2, d1, 0.0),
            u: p(w2, depth, 0.0),
            v: p(w2, d1, h),
        },
        Surface {
            origin: p(w2, d1, 0.0),
            u: p(w1, d1, 0.0),
            v: p(w2, d1, h),
        },
    ]
}

fn uniform_mesh(surfaces: &[Surface], resolution: f64) -> RoomMesh {
    let mut mesh = RoomMesh::new();
    for &surface in surfaces {
        let (nu, nv) = surface_divisions(surface, resolution, None, &[])
            .expect("invalid room mesh resolution");
        add_surface(&mut mesh, surface, nu, nv, false);
    }
    mesh
}

fn adaptive_mesh(
    surfaces: &[Surface],
    base: usize,
    frequency: f64,
    sources: &[Source],
    speed: f64,
) -> RoomMesh {
    let target = if frequency.is_finite() && frequency > 0.0 && speed.is_finite() && speed > 0.0 {
        speed / frequency / 8.0
    } else {
        1.0 / base.max(1) as f64
    };
    let mut mesh = RoomMesh::new();
    for &surface in surfaces {
        let (nu, nv) = surface_divisions(surface, base.max(1) as f64, Some(target), sources)
            .expect("invalid adaptive room mesh resolution");
        add_surface(&mut mesh, surface, nu, nv, true);
    }
    mesh
}

fn add_surface(mesh: &mut RoomMesh, surface: Surface, nu: usize, nv: usize, triangles: bool) {
    let base = mesh.nodes.len();
    for j in 0..=nv {
        for i in 0..=nu {
            let u = if triangles {
                graded(i as f64 / nu as f64)
            } else {
                i as f64 / nu as f64
            };
            let v = if triangles {
                graded(j as f64 / nv as f64)
            } else {
                j as f64 / nv as f64
            };
            mesh.nodes.push(p(
                surface.origin.x
                    + u * (surface.u.x - surface.origin.x)
                    + v * (surface.v.x - surface.origin.x),
                surface.origin.y
                    + u * (surface.u.y - surface.origin.y)
                    + v * (surface.v.y - surface.origin.y),
                surface.origin.z
                    + u * (surface.u.z - surface.origin.z)
                    + v * (surface.v.z - surface.origin.z),
            ));
        }
    }
    for j in 0..nv {
        for i in 0..nu {
            let n0 = base + j * (nu + 1) + i;
            let n1 = n0 + 1;
            let n2 = n0 + nu + 1;
            let n3 = n2 + 1;
            if triangles {
                mesh.elements.push(SurfaceElement::triangle(n0, n1, n2));
                mesh.elements.push(SurfaceElement::triangle(n1, n3, n2));
            } else {
                mesh.elements.push(SurfaceElement::quad(n0, n1, n3, n2));
            }
        }
    }
}

fn distance_to_surface(point: &Point3D, surface: Surface) -> f64 {
    let u = p(
        surface.u.x - surface.origin.x,
        surface.u.y - surface.origin.y,
        surface.u.z - surface.origin.z,
    );
    let v = p(
        surface.v.x - surface.origin.x,
        surface.v.y - surface.origin.y,
        surface.v.z - surface.origin.z,
    );
    let relative = p(
        point.x - surface.origin.x,
        point.y - surface.origin.y,
        point.z - surface.origin.z,
    );
    let uu = u.dot(&u);
    let vv = v.dot(&v);
    if uu < 1e-20 || vv < 1e-20 {
        return point.distance_to(&surface.origin);
    }
    let a = (relative.dot(&u) / uu).clamp(0.0, 1.0);
    let b = (relative.dot(&v) / vv).clamp(0.0, 1.0);
    point.distance_to(&p(
        surface.origin.x + a * u.x + b * v.x,
        surface.origin.y + a * u.y + b * v.y,
        surface.origin.z + a * u.z + b * v.z,
    ))
}

fn graded(t: f64) -> f64 {
    const G: f64 = 0.15;
    if t < G {
        0.5 * (t / G).powi(2) * G
    } else if t > 1.0 - G {
        1.0 - 0.5 * G * ((1.0 - t) / G).powi(2)
    } else {
        t
    }
}

fn p(x: f64, y: f64, z: f64) -> Point3D {
    Point3D::new(x, y, z)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn planned_room_counts_match_generated_meshes() {
        let sources = [Source::omnidirectional(p(0.1, 0.1, 1.0), 1.0)];
        for room in [
            RoomGeometry::Rectangular(RectangularRoom::new(2.0, 3.0, 2.0)),
            RoomGeometry::LShaped(LShapedRoom::new(4.0, 3.0, 2.0, 2.0, 2.0)),
        ] {
            for frequency in [None, Some(50.0), Some(125.0)] {
                let plan = room_surface_mesh_size(&room, 2, frequency, &sources, 343.0).unwrap();
                let mesh = match frequency {
                    Some(f) => room.generate_adaptive_mesh(2, f, &sources, 343.0),
                    None => room.generate_mesh(2),
                };
                assert_eq!(plan.nodes, mesh.nodes.len());
                assert_eq!(plan.elements, mesh.elements.len());
            }
            assert!(room_surface_mesh_size(&room, usize::MAX, None, &[], 343.0).is_err());
            assert!(room_surface_mesh_size(&room, 2, Some(f64::MAX), &[], 343.0).is_err());
        }
    }

    #[test]
    fn exchange_geometries_mesh_through_mesher_extension_trait() {
        let room = RoomGeometry::Rectangular(RectangularRoom::new(2.0, 3.0, 4.0));
        let mesh = room.generate_mesh(2);
        assert!(mesh.num_nodes() > 0 && mesh.num_elements() > 0);
        mesh.validate().unwrap();
    }

    #[test]
    fn l_shaped_adaptive_mesh_refines_for_frequency_and_sources() {
        let room = LShapedRoom::new(4.0, 3.0, 2.0, 2.0, 2.0);
        let base = room.generate_adaptive_mesh(1, 343.0, &[], 343.0);
        let refined = room.generate_adaptive_mesh(1, 686.0, &[], 343.0);
        let source = Source::omnidirectional(p(3.0, 0.1, 1.0), 1.0);
        let near_source = room.generate_adaptive_mesh(1, 343.0, &[source], 343.0);
        assert!(refined.num_elements() > base.num_elements());
        assert!(near_source.num_elements() > base.num_elements());
    }
}
