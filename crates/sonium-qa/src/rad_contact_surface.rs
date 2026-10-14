//! Render explicit quadrilateral contact segments for solid reference meshes.
//!
//! `/SURF/SEG` supports brick faces, unlike the shell-only `/SURF/PART`
//! helper. Callers must establish outward orientation and contact topology;
//! this writer validates identifiers, structural membership and geometry.
//! Format: https://2022.help.altair.com/2022/hwsolvers/rad/topics/solvers/rad/surf_seg_starter_r.htm

// Rust guideline compliant 2026-02-21

use super::{RadError, RadExtendedCase, field_int};
use std::collections::{BTreeMap, BTreeSet};

/// Checked compact geometry and native full-boundary solid self-contact.
#[derive(Debug)]
pub struct NativeSolidContact {
    /// Original one-based deck node IDs in native array order.
    pub node_ids: Vec<usize>,
    /// Structural coordinates; massless rigid-wall drivers are excluded.
    pub coordinates: Vec<[f64; 3]>,
    /// Zero-based brick connectivity in the compact structural array.
    pub bricks: Vec<[usize; 8]>,
    /// Native frictionless sliding penalty operator on the declared boundary.
    pub operator: sonium_fem::physics::solid::contact::explicit::ExplicitContact,
}

/// Build native contact from a checked full-boundary TYPE7 solid bundle.
///
/// Supports one full outward brick boundary, its complete surface-node group,
/// and always-active frictionless constant-gap sliding contact. Unsupported
/// interface flags and additional extra cards are rejected. The supplied
/// base nodal stiffness is multiplied by the declared TYPE7 `Stfac`. This is
/// a native numerical parameter, not a claim of TYPE7
/// stiffness equivalence; its convergence and timestep bound need validation.
///
/// # Errors
/// Rejects invalid cards, incomplete/reversed surfaces, unsupported contact
/// behavior, nonpositive stiffness, or invalid native volume neighborhoods.
pub fn native_solid_self_contact(
    case: &RadExtendedCase,
    surfaces: &[RadSurfSegments],
    extra: &super::RadExtra,
    base_penalty: f64,
) -> Result<NativeSolidContact, RadError> {
    use sonium_fem::physics::solid::contact::explicit::{ExplicitContact, ExplicitContactConfig};
    let invalid = |message: &str| RadError::Parse(format!("native solid contact: {message}"));
    if !base_penalty.is_finite() || base_penalty <= 0.0 || !case.shells.is_empty() {
        return Err(invalid(
            "positive finite stiffness and brick-only geometry required",
        ));
    }
    // Validate the actual external representation before admitting its native
    // counterpart. No contact card may be silently omitted by the adapter.
    write_contact_starter(case, surfaces, extra)?;
    let ([surface], [group], [super::RadInterface::Type7(interface)]) = (
        surfaces,
        extra.grnods.as_slice(),
        extra.interfaces.as_slice(),
    ) else {
        return Err(invalid(
            "exactly one surface, group and TYPE7 interface required",
        ));
    };
    if group.surfs != [surface.id]
        || interface.surf_main != surface.id
        || interface.grnd_secondary != group.id
        || !interface.gapmin_m.is_finite()
        || interface.gapmin_m <= 0.0
        || !interface.stfac.is_finite()
        || interface.stfac <= 0.0
    {
        return Err(invalid(
            "full self-contact references and positive gap/stiffness scale required",
        ));
    }
    let mut supported = super::RadExtra::default();
    supported.grnods.push(group.clone());
    let mut basic =
        super::RadInterType7::basic(interface.id, &interface.title, group.id, surface.id);
    basic.gapmin_m = interface.gapmin_m;
    basic.stfac = interface.stfac;
    supported.interfaces.push(super::RadInterface::Type7(basic));
    if super::write_starter_extra(extra, 1) != super::write_starter_extra(&supported, 1) {
        return Err(invalid(
            "unsupported TYPE7 behavior or additional physical cards",
        ));
    }
    let exterior = brick_exterior(case, surface.id, surface.title.clone())?;
    let canonical = |q: &[usize; 4]| {
        // Cyclic shifts preserve an outward winding; reversal does not.
        (0..4)
            .map(|shift| std::array::from_fn::<_, 4, _>(|j| q[(j + shift) % 4]))
            .min()
            .expect("quadrilateral has four cyclic shifts")
    };
    let expected = exterior
        .quads
        .iter()
        .map(canonical)
        .collect::<BTreeSet<_>>();
    let actual = surface.quads.iter().map(canonical).collect::<BTreeSet<_>>();
    if actual != expected {
        return Err(invalid(
            "every outward brick boundary quad required exactly once",
        ));
    }
    let node_ids = case
        .base
        .mesh
        .bricks
        .iter()
        .flatten()
        .copied()
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>();
    let indices = node_ids
        .iter()
        .enumerate()
        .map(|(i, &id)| (id, i))
        .collect::<BTreeMap<_, _>>();
    let coordinates = node_ids
        .iter()
        .map(|&id| case.base.mesh.nodes[id - 1])
        .collect::<Vec<_>>();
    let bricks = case
        .base
        .mesh
        .bricks
        .iter()
        .map(|b| b.map(|id| indices[&id]))
        .collect::<Vec<_>>();
    let triangles = surface
        .quads
        .iter()
        .flat_map(|q| {
            let q = q.map(|id| indices[&id]);
            [[q[0], q[1], q[2]], [q[0], q[2], q[3]]]
        })
        .collect::<Vec<_>>();
    let nodal_penalty = base_penalty * interface.stfac;
    if !nodal_penalty.is_finite() || nodal_penalty <= 0.0 {
        return Err(invalid("scaled stiffness must remain finite and positive"));
    }
    let operator = ExplicitContact::new_self_solid(
        &triangles,
        &bricks,
        &coordinates,
        ExplicitContactConfig {
            penalty: nodal_penalty,
            friction_coefficient: 0.0,
            search_radius: interface.gapmin_m,
            thickness_offset: interface.gapmin_m,
            ..ExplicitContactConfig::default()
        },
    )
    .map_err(|_| invalid("invalid native solid boundary or volume neighborhood"))?;
    Ok(NativeSolidContact {
        node_ids,
        coordinates,
        bricks,
        operator,
    })
}

/// Render a full starter with checked solid surfaces and contact references.
///
/// Surfaces are emitted before their node groups and interfaces. Extra card
/// references resolve against both explicit segments and part surfaces, in a
/// single surface identifier namespace. Original base fixture cards retain
/// their byte representation. Physical contact behavior requires a solver run.
///
/// # Errors
/// Rejects malformed surfaces, invalid or colliding extra identifiers,
/// dangling contact references, unsupported submodel wrapping, or a missing
/// terminal starter card.
pub fn write_contact_starter(
    case: &RadExtendedCase,
    surfaces: &[RadSurfSegments],
    extra: &super::RadExtra,
) -> Result<String, RadError> {
    if extra.submodel.is_some() {
        return Err(RadError::Parse(
            "explicit contact surface submodels are unsupported".into(),
        ));
    }
    let mut cards = String::new();
    for surface in surfaces {
        cards.push_str(&surface.write_checked(case)?);
    }
    let ids = surfaces.iter().map(|s| s.id).collect::<Vec<_>>();
    extra.validate_with_surface_ids(&super::RadExtraContext::of_extended(case), &ids)?;
    cards.push_str(&super::write_starter_extra(
        extra,
        super::extended_starter_th_groups(case) + 1,
    ));
    let starter = super::write_extended_starter(case);
    let prefix = starter
        .strip_suffix("/END\n")
        .ok_or_else(|| RadError::Parse("missing terminal starter /END".into()))?;
    Ok(format!("{prefix}{cards}/END\n"))
}

/// Render solid contact cards with explicit HEPH defaults and history fields.
///
/// The frozen extended writer omits two solid property rows and writes an
/// unused zero beside every history identifier. The current external reader
/// warns on these omissions and interprets the wall zero as a nonexistent
/// wall. This variant retains formulation24/2 and the actual reader defaults:
/// qa1.1,qb.05,h/lambda/mu0,dtmin0,Istrain1 and default distortion control.
/// Geometry,loads,material,contact and history variable requests are retained.
///
/// # Errors
/// Returns contact validation errors or rejects an unexpected property or
/// history layout rather than silently rewriting arbitrary input cards.
pub fn write_explicit_solid_contact_starter(
    case: &RadExtendedCase,
    surfaces: &[RadSurfSegments],
    extra: &super::RadExtra,
) -> Result<String, RadError> {
    let source = write_contact_starter(case, surfaces, extra)?;
    let lines = source.lines().collect::<Vec<_>>();
    let mut output = String::new();
    let mut index = 0;
    while index < lines.len() {
        if lines[index].starts_with("/PROP/SOLID/") {
            if lines
                .get(index + 2)
                .map(|s| s.split_whitespace().collect::<Vec<_>>())
                != Some(vec!["24", "2"])
                || !lines.get(index + 3).is_some_and(|s| s.starts_with('/'))
            {
                return Err(RadError::Parse(
                    "unexpected frozen HEPH property layout".into(),
                ));
            }
            for line in &lines[index..index + 3] {
                output.push_str(line);
                output.push('\n');
            }
            for value in [1.1, 0.05, 0.0, 0.0, 0.0] {
                output.push_str(&super::field_float(value));
            }
            output.push('\n');
            output.push_str(&super::field_float(0.0));
            output.push_str(&field_int(1));
            output.push_str(&field_int(0));
            output.push('\n');
            index += 3;
            continue;
        }
        if lines[index].starts_with("/TH/BRIC/") || lines[index].starts_with("/TH/RWALL/") {
            if index + 3 > lines.len() {
                return Err(RadError::Parse("incomplete history block".into()));
            }
            for line in &lines[index..index + 3] {
                output.push_str(line);
                output.push('\n');
            }
            index += 3;
            while index < lines.len() && !lines[index].starts_with('/') {
                let fields = lines[index].split_whitespace().collect::<Vec<_>>();
                if fields.len() != 2 || fields[1] != "0" {
                    return Err(RadError::Parse(
                        "unexpected frozen solid/wall history row".into(),
                    ));
                }
                let id = fields[0]
                    .parse::<usize>()
                    .map_err(|_| RadError::Parse("invalid history identifier".into()))?;
                if id == 0 {
                    return Err(RadError::Parse("zero structural history identifier".into()));
                }
                output.push_str(&field_int(id));
                output.push('\n');
                index += 1;
            }
            continue;
        }
        output.push_str(lines[index]);
        output.push('\n');
        index += 1;
    }
    Ok(output)
}

/// Explicit quadrilateral surface with structural node identifiers.
#[derive(Debug, Clone)]
pub struct RadSurfSegments {
    /// Surface identifier, positive and at most ten decimal digits.
    pub id: usize,
    /// Single ASCII title line, at most one hundred characters.
    pub title: String,
    /// Ordered quadrilateral node identifiers, one-based in the case mesh.
    pub quads: Vec<[usize; 4]>,
}

fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// Extract outward brick boundary faces without retaining internal interfaces.
///
/// Shared faces are identified by their four node identifiers. Each retained
/// face is oriented away from its brick centroid. This supports hollow solid
/// meshes, including inner cavity faces, without treating internal element
/// interfaces as physical contact surfaces.
///
/// # Errors
/// Rejects invalid connectivity, nonfinite nodes, nonmanifold face incidence,
/// an empty boundary, or geometry rejected by the checked surface writer.
pub fn brick_exterior(
    case: &RadExtendedCase,
    id: usize,
    title: String,
) -> Result<RadSurfSegments, RadError> {
    let mut faces = BTreeMap::<[usize; 4], (usize, [usize; 4])>::new();
    // Standard eight-node brick face connectivity; orientation is checked
    // geometrically below rather than assuming the input winding.
    let local = [
        [0, 1, 2, 3],
        [4, 5, 6, 7],
        [0, 1, 5, 4],
        [1, 2, 6, 5],
        [2, 3, 7, 6],
        [3, 0, 4, 7],
    ];
    for brick in &case.base.mesh.bricks {
        if brick
            .iter()
            .any(|n| *n == 0 || *n > case.base.mesh.nodes.len())
        {
            return Err(RadError::Parse(
                "contact boundary has invalid brick node".into(),
            ));
        }
        let points = brick.map(|n| case.base.mesh.nodes[n - 1]);
        if points.iter().flatten().any(|v| !v.is_finite()) {
            return Err(RadError::Parse(
                "contact boundary has nonfinite brick coordinate".into(),
            ));
        }
        // Relative coordinates avoid introducing a large world-origin offset.
        let center: [f64; 3] = std::array::from_fn(|axis| {
            points
                .iter()
                .map(|p| p[axis] - points[0][axis])
                .sum::<f64>()
                / 8.0
        });
        for local_face in local {
            let mut face = local_face.map(|i| brick[i]);
            let p = local_face.map(|i| points[i]);
            let edge = |i: usize| std::array::from_fn(|axis| p[i][axis] - p[0][axis]);
            let normal = cross(edge(1), edge(2));
            let outward: [f64; 3] = std::array::from_fn(|axis| {
                p.iter().map(|p| p[axis] - points[0][axis]).sum::<f64>() / 4.0 - center[axis]
            });
            let alignment = normal.iter().zip(outward).map(|(a, b)| a * b).sum::<f64>();
            if !alignment.is_finite() || alignment == 0.0 {
                return Err(RadError::Parse(
                    "cannot orient collapsed brick boundary".into(),
                ));
            }
            if alignment < 0.0 {
                face = [face[0], face[3], face[2], face[1]];
            }
            let mut key = face;
            key.sort_unstable();
            let entry = faces.entry(key).or_insert((0, face));
            entry.0 += 1;
            if entry.0 > 2 {
                return Err(RadError::Parse("nonmanifold brick contact boundary".into()));
            }
        }
    }
    let surface = RadSurfSegments {
        id,
        title,
        quads: faces
            .into_values()
            .filter_map(|(count, face)| (count == 1).then_some(face))
            .collect(),
    };
    surface.write_checked(case)?;
    Ok(surface)
}

impl RadSurfSegments {
    /// Render a checked `/SURF/SEG` card for an existing reference case.
    ///
    /// Each node must belong to a brick or shell element. Segment identifiers
    /// are generated from one; node ordering is retained exactly. The output
    /// belongs before the starter's `/END` card. This checks connectivity and
    /// nondegenerate geometry, not outward orientation, surface closure or
    /// physical contact convergence. Triangular segments are not supported.
    ///
    /// # Errors
    /// Rejects invalid identifiers or titles, empty or duplicate segments,
    /// nonstructural nodes, nonfinite coordinates, collapsed or folded faces.
    pub fn write_checked(&self, case: &RadExtendedCase) -> Result<String, RadError> {
        let invalid = |message: &str| RadError::Parse(format!("/SURF/SEG: {message}"));
        if self.id == 0
            || self.id as u64 > 9_999_999_999
            || self.quads.is_empty()
            || self.title.is_empty()
            || self.title.len() > 100
            || !self.title.is_ascii()
            || self.title.contains(['\r', '\n'])
        {
            return Err(invalid(
                "positive ten-digit id,nonempty ASCII title and segments required",
            ));
        }
        let structural = case
            .base
            .mesh
            .bricks
            .iter()
            .flatten()
            .chain(case.shells.quads.iter().flatten())
            .chain(case.shells.tris.iter().flatten())
            .copied()
            .collect::<BTreeSet<_>>();
        let mut seen = BTreeSet::new();
        for quad in &self.quads {
            let mut key = *quad;
            key.sort_unstable();
            if key.windows(2).any(|v| v[0] == v[1]) || !seen.insert(key) {
                return Err(invalid("repeated nodes or duplicate quadrilateral"));
            }
            if quad.iter().any(|id| {
                *id == 0
                    || *id > case.base.mesh.nodes.len()
                    || *id as u64 > 9_999_999_999
                    || !structural.contains(id)
            }) {
                return Err(invalid("segment node must belong to a structural element"));
            }
            let points = quad.map(|id| case.base.mesh.nodes[id - 1]);
            if points.iter().flatten().any(|v| !v.is_finite()) {
                return Err(invalid("nonfinite segment coordinate"));
            }
            let edge =
                |node: usize| std::array::from_fn(|axis| points[node][axis] - points[0][axis]);
            let a = cross(edge(1), edge(2));
            let b = cross(edge(2), edge(3));
            let alignment = a.iter().zip(b).map(|(a, b)| a * b).sum::<f64>();
            if !alignment.is_finite() || alignment <= 0.0 {
                return Err(invalid("collapsed or folded quadrilateral"));
            }
        }
        let mut deck = format!("/SURF/SEG/{}/1\n{}\n", self.id, self.title);
        for (index, quad) in self.quads.iter().enumerate() {
            deck.push_str(&field_int(index + 1));
            for &node in quad {
                deck.push_str(&field_int(node));
            }
            deck.push('\n');
        }
        Ok(deck)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn native_fixture() -> (RadExtendedCase, RadSurfSegments, super::super::RadExtra) {
        let (mut case, _) = fixture();
        case.base.mesh.add_node([0.0; 3]); // Unreferenced massless driver.
        let surface = brick_exterior(&case, 100, "full solid boundary".into()).unwrap();
        let mut extra = super::super::RadExtra::default();
        extra.grnods.push(super::super::RadGrnodSurfs {
            id: 200,
            title: "all surface nodes".into(),
            surfs: vec![100],
        });
        let mut interface = super::super::RadInterType7::basic(300, "self contact", 200, 100);
        interface.gapmin_m = 0.0005;
        extra
            .interfaces
            .push(super::super::RadInterface::Type7(interface));
        (case, surface, extra)
    }

    #[test]
    fn native_adapter_compacts_drivers_and_applies_declared_contact_scale() {
        let (case, surface, extra) = native_fixture();
        let mut native = native_solid_self_contact(&case, &[surface], &extra, 1000.0).unwrap();
        assert!(!native.node_ids.contains(&case.base.mesh.nodes.len()));
        assert_eq!(native.operator.config().penalty, 200.0);
        assert_eq!(native.operator.config().thickness_offset, 0.0005);
        let coords = native
            .coordinates
            .iter()
            .flatten()
            .copied()
            .collect::<Vec<_>>();
        let mut forces = vec![0.0; coords.len()];
        let pairs =
            native
                .operator
                .apply_with_pairs(&coords, &vec![0.0; coords.len()], &mut forces);
        assert!(pairs.is_empty());
        assert!(forces.iter().all(|f| *f == 0.0));
        for (i, &id) in native.node_ids.iter().enumerate() {
            assert_eq!(native.coordinates[i], case.base.mesh.nodes[id - 1]);
        }
    }

    #[test]
    fn native_adapter_rejects_unsupported_physics_and_surface_winding() {
        let (case, surface, extra) = native_fixture();
        for tied in [false, true] {
            let mut unsupported = extra.clone();
            let super::super::RadInterface::Type7(interface) = &mut unsupported.interfaces[0]
            else {
                unreachable!();
            };
            if tied {
                interface.itied = 1;
            } else {
                interface.fric = 0.1;
            }
            assert!(
                native_solid_self_contact(&case, std::slice::from_ref(&surface), &unsupported, 1.0)
                    .is_err()
            );
        }
        let mut reversed = surface.clone();
        reversed.quads[0].reverse();
        assert!(native_solid_self_contact(&case, &[reversed], &extra, 1.0).is_err());
        let mut partial = surface;
        partial.quads.pop();
        assert!(native_solid_self_contact(&case, &[partial], &extra, 1.0).is_err());
    }

    fn fixture() -> (RadExtendedCase, RadSurfSegments) {
        let case = RadExtendedCase::wrap_elastic(super::super::cantilever_case());
        let b = case.base.mesh.bricks[0];
        let surface = RadSurfSegments {
            id: 100,
            title: "solid face".into(),
            quads: vec![[b[0], b[3], b[2], b[1]]],
        };
        (case, surface)
    }

    #[test]
    fn solid_face_card_preserves_documented_field_order_and_winding() {
        let (case, surface) = fixture();
        let text = surface.write_checked(&case).unwrap();
        let mut lines = text.lines();
        assert_eq!(lines.next(), Some("/SURF/SEG/100/1"));
        assert_eq!(lines.next(), Some("solid face"));
        let row = lines.next().unwrap();
        assert_eq!(row.len(), 50);
        let fields = row
            .split_whitespace()
            .map(|s| s.parse::<usize>().unwrap())
            .collect::<Vec<_>>();
        assert_eq!(fields, [vec![1], surface.quads[0].to_vec()].concat());
        assert!(lines.next().is_none());
    }

    #[test]
    fn malformed_contact_topology_and_massless_driver_are_rejected() {
        let (mut case, mut surface) = fixture();
        let driver = case.base.mesh.add_node([1.0, 1.0, 1.0]);
        surface.quads[0][0] = driver;
        assert!(surface.write_checked(&case).is_err());
        let (mut case, mut surface) = fixture();
        surface.quads.push(surface.quads[0]);
        assert!(surface.write_checked(&case).is_err());
        surface.quads.pop();
        let id = surface.quads[0][1];
        case.base.mesh.nodes[id - 1] = case.base.mesh.nodes[surface.quads[0][0] - 1];
        assert!(surface.write_checked(&case).is_err());
        let (case, mut surface) = fixture();
        surface.title = "injected\n/END".into();
        assert!(surface.write_checked(&case).is_err());
    }

    #[test]
    fn extracted_boundary_has_analytic_area_and_volume_without_internal_faces() {
        let (case, _) = fixture();
        let surface = brick_exterior(&case, 100, "complete solid exterior".into()).unwrap();
        // Ten by two by two bricks:2*(10*2+10*2+2*2) boundary faces.
        assert_eq!(surface.quads.len(), 88);
        let mut area = 0.0;
        let mut volume = 0.0;
        for face in surface.quads {
            let p = face.map(|n| case.base.mesh.nodes[n - 1]);
            for triangle in [[0, 1, 2], [0, 2, 3]] {
                let a = p[triangle[0]];
                let b = p[triangle[1]];
                let c = p[triangle[2]];
                let ab = std::array::from_fn(|i| b[i] - a[i]);
                let ac = std::array::from_fn(|i| c[i] - a[i]);
                let n = cross(ab, ac);
                area += n.iter().map(|v| v * v).sum::<f64>().sqrt() / 2.0;
                volume += a.iter().zip(cross(b, c)).map(|(a, n)| a * n).sum::<f64>() / 6.0;
            }
        }
        assert!((area - 0.205).abs() < 1e-13);
        assert!((volume - 0.0025).abs() < 1e-14);
    }

    #[test]
    fn solid_contact_bundle_resolves_surface_groups_and_rejects_id_collisions() {
        let (case, surface) = fixture();
        let mut extra = super::super::RadExtra::default();
        extra.grnods.push(super::super::RadGrnodSurfs {
            id: 200,
            title: "surface nodes".into(),
            surfs: vec![100],
        });
        extra.interfaces.push(super::super::RadInterface::Type7(
            super::super::RadInterType7::basic(300, "self contact", 200, 100),
        ));
        let text = write_contact_starter(&case, std::slice::from_ref(&surface), &extra).unwrap();
        assert!(text.find("/SURF/SEG/100/1").unwrap() < text.find("/GRNOD/SURF/200").unwrap());
        assert!(text.find("/GRNOD/SURF/200").unwrap() < text.find("/INTER/TYPE7/300").unwrap());
        assert!(text.ends_with("/END\n"));
        assert!(write_contact_starter(&case, &[], &extra).is_err());
        extra.surfs.push(super::super::RadSurfParts {
            id: 100,
            title: "collision".into(),
            parts: vec![1],
        });
        assert!(write_contact_starter(&case, &[surface], &extra).is_err());
        let default =
            write_contact_starter(&case, &[], &super::super::RadExtra::default()).unwrap();
        assert_eq!(default, super::super::write_extended_starter(&case));
    }

    #[test]
    fn explicit_solid_layout_retains_defaults_and_all_history_identifiers() {
        let (mut case, surface) = fixture();
        case.th_brick = super::super::RadThRequest::brick_stress(vec![1, 2]);
        let text = write_explicit_solid_contact_starter(
            &case,
            &[surface],
            &super::super::RadExtra::default(),
        )
        .unwrap();
        let rows = text.lines().collect::<Vec<_>>();
        let property = rows
            .iter()
            .position(|s| s.starts_with("/PROP/SOLID/"))
            .unwrap();
        assert_eq!(
            rows[property + 2].split_whitespace().collect::<Vec<_>>(),
            ["24", "2"]
        );
        let parameters = rows[property + 3]
            .split_whitespace()
            .map(|s| s.parse::<f64>().unwrap())
            .collect::<Vec<_>>();
        assert_eq!(parameters, [1.1, 0.05, 0.0, 0.0, 0.0]);
        let history = rows
            .iter()
            .position(|s| s.starts_with("/TH/BRIC/"))
            .unwrap();
        assert_eq!(
            rows[history + 3].split_whitespace().collect::<Vec<_>>(),
            ["1"]
        );
        assert_eq!(
            rows[history + 4].split_whitespace().collect::<Vec<_>>(),
            ["2"]
        );
    }
}
