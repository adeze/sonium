//! Small planar triangulation helpers.

use std::collections::HashSet;

use super::{BoundaryType, ElementType, Mesh, Point};

/// Failure while triangulating a simple planar polygon.
#[derive(Debug, Clone, PartialEq)]
pub enum PolygonTriangulationError {
    /// Fewer than three distinct vertices were supplied.
    TooFewVertices,
    /// A coordinate is NaN or infinite.
    NonFiniteVertex { index: usize },
    /// The polygon has zero signed area.
    DegeneratePolygon,
    /// Ear clipping could not find a valid ear, normally indicating a
    /// self-intersection or repeated non-adjacent vertex.
    NonSimplePolygon,
    /// Uniform mesh refinement rejected the generated topology.
    Refinement(String),
}

impl std::fmt::Display for PolygonTriangulationError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::TooFewVertices => write!(formatter, "a polygon needs at least three vertices"),
            Self::NonFiniteVertex { index } => {
                write!(formatter, "polygon vertex {index} is not finite")
            }
            Self::DegeneratePolygon => write!(formatter, "polygon has zero signed area"),
            Self::NonSimplePolygon => write!(formatter, "polygon is not simple"),
            Self::Refinement(reason) => write!(formatter, "polygon refinement failed: {reason}"),
        }
    }
}

impl std::error::Error for PolygonTriangulationError {}

/// Triangulate a counter-clockwise or clockwise simple polygon by ear clipping.
///
/// The optional repeated closing vertex is accepted. Returned indices refer to
/// the polygon after removing that closing duplicate and triangles are CCW.
pub fn triangulate_simple_polygon(
    vertices: &[(f64, f64)],
) -> Result<Vec<[usize; 3]>, PolygonTriangulationError> {
    let mut count = vertices.len();
    if count >= 2 && vertices[0] == vertices[count - 1] {
        count -= 1;
    }
    if count < 3 {
        return Err(PolygonTriangulationError::TooFewVertices);
    }
    if let Some(index) = vertices[..count]
        .iter()
        .position(|&(x, y)| !x.is_finite() || !y.is_finite())
    {
        return Err(PolygonTriangulationError::NonFiniteVertex { index });
    }
    let area_twice: f64 = (0..count)
        .map(|index| {
            let next = (index + 1) % count;
            vertices[index].0 * vertices[next].1 - vertices[next].0 * vertices[index].1
        })
        .sum();
    if area_twice.abs() <= f64::EPSILON {
        return Err(PolygonTriangulationError::DegeneratePolygon);
    }
    let mut remaining: Vec<usize> = if area_twice > 0.0 {
        (0..count).collect()
    } else {
        (0..count).rev().collect()
    };
    let scale = vertices[..count]
        .iter()
        .fold(1.0_f64, |value, &(x, y)| value.max(x.abs()).max(y.abs()));
    let tolerance = 64.0 * f64::EPSILON * scale * scale;
    let mut triangles = Vec::with_capacity(count - 2);
    while remaining.len() > 3 {
        let mut clipped = false;
        for position in 0..remaining.len() {
            let previous = remaining[(position + remaining.len() - 1) % remaining.len()];
            let current = remaining[position];
            let next = remaining[(position + 1) % remaining.len()];
            if orient(vertices[previous], vertices[current], vertices[next]) <= tolerance {
                continue;
            }
            let contains_vertex = remaining.iter().any(|&candidate| {
                candidate != previous
                    && candidate != current
                    && candidate != next
                    && point_in_ccw_triangle(
                        vertices[candidate],
                        vertices[previous],
                        vertices[current],
                        vertices[next],
                        tolerance,
                    )
            });
            if contains_vertex {
                continue;
            }
            triangles.push([previous, current, next]);
            remaining.remove(position);
            clipped = true;
            break;
        }
        if !clipped {
            return Err(PolygonTriangulationError::NonSimplePolygon);
        }
    }
    triangles.push([remaining[0], remaining[1], remaining[2]]);
    Ok(triangles)
}

/// Triangulate a convex outer polygon containing zero or more polygonal holes.
///
/// This is a constrained-Delaunay admission wrapper for the common PLC case:
/// all loop vertices are triangulated together, triangles outside the outer
/// loop or inside a hole are discarded, and the result is accepted only if
/// every supplied loop edge survives unchanged as a mesh boundary edge.  It
/// therefore fails closed rather than returning a mesh whose triangles cut
/// across a circular rim or another hole.  The outer loop must be convex;
/// callers needing a non-convex exterior can decompose it into convex panels
/// before invoking this helper.
///
/// Returned vertices consist of the outer loop followed by each hole loop,
/// with an optional repeated closing vertex removed from every loop. Returned
/// triangles are counter-clockwise and index that flattened vertex array.
type PolygonTriangulation = Result<(Vec<(f64, f64)>, Vec<[usize; 3]>), PolygonTriangulationError>;

pub fn triangulate_convex_polygon_with_holes(
    outer: &[(f64, f64)],
    holes: &[&[(f64, f64)]],
) -> PolygonTriangulation {
    let outer = normalized_polygon_loop(outer)?;
    if !is_convex_loop(&outer) {
        return Err(PolygonTriangulationError::NonSimplePolygon);
    }

    let mut vertices = outer.clone();
    let mut loops = vec![(0..outer.len()).collect::<Vec<_>>()];
    for hole in holes {
        let hole = normalized_polygon_loop(hole)?;
        if !point_in_polygon(hole[0], &outer) || loop_edges_intersect(&hole, &outer) {
            return Err(PolygonTriangulationError::NonSimplePolygon);
        }
        let start = vertices.len();
        vertices.extend(hole);
        loops.push((start..vertices.len()).collect());
    }
    if vertices.iter().enumerate().any(|(index, point)| {
        vertices[..index]
            .iter()
            .any(|previous| same_point(*point, *previous))
    }) {
        return Err(PolygonTriangulationError::NonSimplePolygon);
    }
    for first in 1..loops.len() {
        for second in 1..first {
            let first_loop = loop_points(&vertices, &loops[first]);
            let second_loop = loop_points(&vertices, &loops[second]);
            if point_in_polygon(first_loop[0], &second_loop)
                || point_in_polygon(second_loop[0], &first_loop)
                || loop_edges_intersect(&first_loop, &second_loop)
            {
                return Err(PolygonTriangulationError::NonSimplePolygon);
            }
        }
    }

    let mut candidate_triangles = triangulate_points_2d(&vertices);
    if !recover_loop_edges(&vertices, &mut candidate_triangles, &loops) {
        return Err(PolygonTriangulationError::NonSimplePolygon);
    }
    let mut triangles = Vec::new();
    for mut triangle in candidate_triangles {
        let centroid = triangle_centroid(&vertices, triangle);
        if !point_in_polygon(centroid, &outer)
            || loops[1..]
                .iter()
                .map(|loop_indices| loop_points(&vertices, loop_indices))
                .any(|hole| point_in_polygon(centroid, &hole))
        {
            continue;
        }
        if orient(
            vertices[triangle[0]],
            vertices[triangle[1]],
            vertices[triangle[2]],
        ) < 0.0
        {
            triangle.swap(1, 2);
        }
        triangles.push(triangle);
    }

    let triangle_edges = triangles
        .iter()
        .flat_map(|triangle| {
            [
                sorted_edge(triangle[0], triangle[1]),
                sorted_edge(triangle[1], triangle[2]),
                sorted_edge(triangle[2], triangle[0]),
            ]
        })
        .collect::<HashSet<_>>();
    if !loop_edges_present(&triangle_edges, &loops) {
        return Err(PolygonTriangulationError::NonSimplePolygon);
    }
    Ok((vertices, triangles))
}

/// Build a native 2D FEM mesh for a simple polygon and uniformly refine it.
pub fn polygon_mesh_triangles(
    vertices: &[(f64, f64)],
    refinement_levels: usize,
    boundary_marker: i32,
) -> Result<Mesh, PolygonTriangulationError> {
    let closed = vertices.len() >= 2 && vertices[0] == vertices[vertices.len() - 1];
    let vertex_count = vertices.len() - usize::from(closed);
    let triangles = triangulate_simple_polygon(vertices)?;
    let mut mesh = Mesh::new(2);
    for &(x, y) in &vertices[..vertex_count] {
        mesh.add_node(Point::new_2d(x, y));
    }
    for triangle in triangles {
        mesh.add_element(ElementType::Triangle, triangle.to_vec());
    }
    mesh.detect_boundaries();
    for boundary in &mut mesh.boundaries {
        boundary.boundary_type = BoundaryType::Dirichlet;
        boundary.marker = boundary_marker;
    }
    for _ in 0..refinement_levels {
        super::refinement::uniform_refine_checked(&mut mesh)
            .map_err(|error| PolygonTriangulationError::Refinement(error.to_string()))?;
    }
    Ok(mesh)
}

fn orient(a: (f64, f64), b: (f64, f64), c: (f64, f64)) -> f64 {
    (b.0 - a.0) * (c.1 - a.1) - (b.1 - a.1) * (c.0 - a.0)
}

fn normalized_polygon_loop(
    vertices: &[(f64, f64)],
) -> Result<Vec<(f64, f64)>, PolygonTriangulationError> {
    let count = vertices.len()
        - usize::from(vertices.len() >= 2 && vertices[0] == vertices[vertices.len() - 1]);
    if count < 3 {
        return Err(PolygonTriangulationError::TooFewVertices);
    }
    let vertices = vertices[..count].to_vec();
    if let Some(index) = vertices
        .iter()
        .position(|&(x, y)| !x.is_finite() || !y.is_finite())
    {
        return Err(PolygonTriangulationError::NonFiniteVertex { index });
    }
    if signed_area_twice(&vertices).abs() <= f64::EPSILON {
        return Err(PolygonTriangulationError::DegeneratePolygon);
    }
    Ok(vertices)
}

fn signed_area_twice(vertices: &[(f64, f64)]) -> f64 {
    (0..vertices.len())
        .map(|index| {
            let next = (index + 1) % vertices.len();
            vertices[index].0 * vertices[next].1 - vertices[next].0 * vertices[index].1
        })
        .sum()
}

fn is_convex_loop(vertices: &[(f64, f64)]) -> bool {
    let orientation = signed_area_twice(vertices).signum();
    (0..vertices.len()).all(|index| {
        let previous = vertices[(index + vertices.len() - 1) % vertices.len()];
        let current = vertices[index];
        let next = vertices[(index + 1) % vertices.len()];
        orientation * orient(previous, current, next) >= -f64::EPSILON
    })
}

fn loop_points(vertices: &[(f64, f64)], indices: &[usize]) -> Vec<(f64, f64)> {
    indices.iter().map(|&index| vertices[index]).collect()
}

fn point_in_polygon(point: (f64, f64), polygon: &[(f64, f64)]) -> bool {
    let mut inside = false;
    for index in 0..polygon.len() {
        let a = polygon[index];
        let b = polygon[(index + 1) % polygon.len()];
        if ((a.1 > point.1) != (b.1 > point.1))
            && point.0 < (b.0 - a.0) * (point.1 - a.1) / (b.1 - a.1) + a.0
        {
            inside = !inside;
        }
    }
    inside
}

fn same_point(a: (f64, f64), b: (f64, f64)) -> bool {
    (a.0 - b.0).abs() <= 1.0e-12 && (a.1 - b.1).abs() <= 1.0e-12
}

fn loop_edges_intersect(first: &[(f64, f64)], second: &[(f64, f64)]) -> bool {
    (0..first.len()).any(|first_index| {
        let a = first[first_index];
        let b = first[(first_index + 1) % first.len()];
        (0..second.len()).any(|second_index| {
            let c = second[second_index];
            let d = second[(second_index + 1) % second.len()];
            segments_intersect(a, b, c, d)
        })
    })
}

fn segments_intersect(a: (f64, f64), b: (f64, f64), c: (f64, f64), d: (f64, f64)) -> bool {
    let ab_c = orient(a, b, c);
    let ab_d = orient(a, b, d);
    let cd_a = orient(c, d, a);
    let cd_b = orient(c, d, b);
    (ab_c.abs() <= 1.0e-12 && point_on_segment(c, a, b))
        || (ab_d.abs() <= 1.0e-12 && point_on_segment(d, a, b))
        || (cd_a.abs() <= 1.0e-12 && point_on_segment(a, c, d))
        || (cd_b.abs() <= 1.0e-12 && point_on_segment(b, c, d))
        || (ab_c * ab_d < 0.0 && cd_a * cd_b < 0.0)
}

fn point_on_segment(point: (f64, f64), start: (f64, f64), end: (f64, f64)) -> bool {
    point.0 >= start.0.min(end.0) - 1.0e-12
        && point.0 <= start.0.max(end.0) + 1.0e-12
        && point.1 >= start.1.min(end.1) - 1.0e-12
        && point.1 <= start.1.max(end.1) + 1.0e-12
}

fn triangle_centroid(vertices: &[(f64, f64)], triangle: [usize; 3]) -> (f64, f64) {
    let [a, b, c] = triangle.map(|index| vertices[index]);
    ((a.0 + b.0 + c.0) / 3.0, (a.1 + b.1 + c.1) / 3.0)
}

fn sorted_edge(first: usize, second: usize) -> [usize; 2] {
    [first.min(second), first.max(second)]
}

fn loop_edges_present(edges: &HashSet<[usize; 2]>, loops: &[Vec<usize>]) -> bool {
    loops.iter().all(|loop_indices| {
        loop_indices.iter().enumerate().all(|(index, &node)| {
            edges.contains(&sorted_edge(
                node,
                loop_indices[(index + 1) % loop_indices.len()],
            ))
        })
    })
}

/// Recover planar constraint edges through local diagonal flips.  The supplied
/// loops are pairwise disjoint, so a recovered edge is never allowed to be
/// flipped again while recovering a later constraint.
#[allow(clippy::ptr_arg)]
fn recover_loop_edges(
    vertices: &[(f64, f64)],
    triangles: &mut Vec<[usize; 3]>,
    loops: &[Vec<usize>],
) -> bool {
    let constraints = loops
        .iter()
        .flat_map(|loop_indices| {
            loop_indices.iter().enumerate().map(|(index, &node)| {
                sorted_edge(node, loop_indices[(index + 1) % loop_indices.len()])
            })
        })
        .collect::<Vec<_>>();
    let mut recovered = HashSet::new();
    let budget = constraints
        .len()
        .saturating_mul(triangles.len())
        .saturating_mul(2);
    let mut flips = 0;
    for constraint in constraints {
        while !mesh_edge_set(triangles).contains(&constraint) {
            if flips >= budget {
                return false;
            }
            let Some((edge, owners)) =
                crossing_interior_edge(vertices, triangles, constraint, &recovered)
            else {
                return false;
            };
            let [first_owner, second_owner] = owners;
            let Some(first_opposite) = triangle_opposite(triangles[first_owner], edge) else {
                return false;
            };
            let Some(second_opposite) = triangle_opposite(triangles[second_owner], edge) else {
                return false;
            };
            let replacement = sorted_edge(first_opposite, second_opposite);
            if recovered.contains(&replacement)
                || !strict_segments_intersect(
                    vertices[edge[0]],
                    vertices[edge[1]],
                    vertices[first_opposite],
                    vertices[second_opposite],
                )
            {
                return false;
            }
            triangles[first_owner] =
                oriented_triangle(edge[0], first_opposite, second_opposite, vertices);
            triangles[second_owner] =
                oriented_triangle(edge[1], second_opposite, first_opposite, vertices);
            flips += 1;
        }
        recovered.insert(constraint);
    }
    true
}

fn mesh_edge_set(triangles: &[[usize; 3]]) -> HashSet<[usize; 2]> {
    triangles
        .iter()
        .flat_map(|triangle| {
            [
                sorted_edge(triangle[0], triangle[1]),
                sorted_edge(triangle[1], triangle[2]),
                sorted_edge(triangle[2], triangle[0]),
            ]
        })
        .collect()
}

fn crossing_interior_edge(
    vertices: &[(f64, f64)],
    triangles: &[[usize; 3]],
    constraint: [usize; 2],
    recovered: &HashSet<[usize; 2]>,
) -> Option<([usize; 2], [usize; 2])> {
    let mut owners = std::collections::BTreeMap::<[usize; 2], Vec<usize>>::new();
    for (triangle_index, triangle) in triangles.iter().enumerate() {
        for edge in [
            sorted_edge(triangle[0], triangle[1]),
            sorted_edge(triangle[1], triangle[2]),
            sorted_edge(triangle[2], triangle[0]),
        ] {
            owners.entry(edge).or_default().push(triangle_index);
        }
    }
    owners.into_iter().find_map(|(edge, owners)| {
        (owners.len() == 2
            && !recovered.contains(&edge)
            && edge[0] != constraint[0]
            && edge[0] != constraint[1]
            && edge[1] != constraint[0]
            && edge[1] != constraint[1]
            && strict_segments_intersect(
                vertices[edge[0]],
                vertices[edge[1]],
                vertices[constraint[0]],
                vertices[constraint[1]],
            ))
        .then_some((edge, [owners[0], owners[1]]))
    })
}

fn strict_segments_intersect(a: (f64, f64), b: (f64, f64), c: (f64, f64), d: (f64, f64)) -> bool {
    let ab_c = orient(a, b, c);
    let ab_d = orient(a, b, d);
    let cd_a = orient(c, d, a);
    let cd_b = orient(c, d, b);
    ab_c * ab_d < -1.0e-24 && cd_a * cd_b < -1.0e-24
}

fn triangle_opposite(triangle: [usize; 3], edge: [usize; 2]) -> Option<usize> {
    triangle
        .into_iter()
        .find(|&vertex| vertex != edge[0] && vertex != edge[1])
}

fn oriented_triangle(a: usize, b: usize, c: usize, vertices: &[(f64, f64)]) -> [usize; 3] {
    if orient(vertices[a], vertices[b], vertices[c]) > 0.0 {
        [a, b, c]
    } else {
        [a, c, b]
    }
}

fn point_in_ccw_triangle(
    point: (f64, f64),
    a: (f64, f64),
    b: (f64, f64),
    c: (f64, f64),
    tolerance: f64,
) -> bool {
    orient(a, b, point) >= -tolerance
        && orient(b, c, point) >= -tolerance
        && orient(c, a, point) >= -tolerance
}

/// Triangulate arbitrary 2D points with Delaunay connectivity.
///
/// The returned triangles index into `points`. Degenerate triangles are
/// filtered out, which also keeps collinear and duplicate-heavy inputs from
/// leaking unusable elements to callers.
pub fn triangulate_points_2d(points: &[(f64, f64)]) -> Vec<[usize; 3]> {
    if points.len() < 3 {
        return Vec::new();
    }

    let delaunay_points: Vec<delaunator::Point> = points
        .iter()
        .map(|&(x, y)| delaunator::Point { x, y })
        .collect();
    let triangulation = delaunator::triangulate(&delaunay_points);

    triangulation
        .triangles
        .as_chunks::<3>()
        .0
        .iter()
        .filter_map(|tri| {
            let triangle = [tri[0], tri[1], tri[2]];
            is_valid_triangle(points, triangle).then_some(oriented_triangle(
                triangle[0],
                triangle[1],
                triangle[2],
                points,
            ))
        })
        .collect()
}

fn is_valid_triangle(points: &[(f64, f64)], tri: [usize; 3]) -> bool {
    let n = points.len();
    if tri.iter().any(|&idx| idx >= n) || tri[0] == tri[1] || tri[1] == tri[2] || tri[2] == tri[0] {
        return false;
    }

    let [a, b, c] = tri;
    let (ax, ay) = points[a];
    let (bx, by) = points[b];
    let (cx, cy) = points[c];
    let area2 = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax);
    area2.is_finite() && area2.abs() > f64::EPSILON
}

#[cfg(test)]
mod tests {
    use super::{
        orient, polygon_mesh_triangles, triangulate_convex_polygon_with_holes,
        triangulate_points_2d, triangulate_simple_polygon,
    };

    #[test]
    fn square_produces_two_triangles() {
        let points = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)];
        let triangles = triangulate_points_2d(&points);
        assert_eq!(triangles.len(), 2);
        assert!(triangles.iter().flatten().all(|&idx| idx < points.len()));
        assert!(
            triangles
                .iter()
                .all(|&[a, b, c]| { orient(points[a], points[b], points[c]) > 0.0 })
        );
    }

    #[test]
    fn short_inputs_have_no_triangles() {
        assert!(triangulate_points_2d(&[]).is_empty());
        assert!(triangulate_points_2d(&[(0.0, 0.0), (1.0, 0.0)]).is_empty());
    }

    #[test]
    fn convex_panel_with_two_holes_preserves_every_loop_edge() {
        let outer = [(-0.3, -0.2), (0.3, -0.2), (0.3, 0.2), (-0.3, 0.2)];
        let first = [(-0.18, -0.05), (-0.08, -0.05), (-0.08, 0.05), (-0.18, 0.05)];
        let second = [(0.08, -0.05), (0.18, -0.05), (0.18, 0.05), (0.08, 0.05)];
        let (vertices, triangles) =
            triangulate_convex_polygon_with_holes(&outer, &[&first, &second])
                .expect("two disjoint panel holes triangulate");

        let triangle_area = triangles
            .iter()
            .map(|&[a, b, c]| {
                ((vertices[b].0 - vertices[a].0) * (vertices[c].1 - vertices[a].1)
                    - (vertices[b].1 - vertices[a].1) * (vertices[c].0 - vertices[a].0))
                    * 0.5
            })
            .sum::<f64>();
        let expected_area = 0.6 * 0.4 - 2.0 * 0.1 * 0.1;
        assert!((triangle_area - expected_area).abs() < 1.0e-12);
        assert!(triangles.iter().all(|&[a, b, c]| {
            let centroid = (
                (vertices[a].0 + vertices[b].0 + vertices[c].0) / 3.0,
                (vertices[a].1 + vertices[b].1 + vertices[c].1) / 3.0,
            );
            !(((-0.18..-0.08).contains(&centroid.0) || (0.08..0.18).contains(&centroid.0))
                && (-0.05..0.05).contains(&centroid.1))
        }));
    }

    #[test]
    fn subdivided_rectangular_panel_with_circular_holes_triangulates() {
        let outer = [(-0.2, -0.1), (0.2, -0.1), (0.2, 0.1), (-0.2, 0.1)];
        let ring = |center_x| {
            (0..24)
                .map(|index| {
                    let angle = std::f64::consts::TAU * index as f64 / 24.0;
                    (center_x + 0.025 * angle.cos(), 0.025 * angle.sin())
                })
                .collect::<Vec<_>>()
        };
        let first = ring(-0.1);
        let second = ring(0.1);
        triangulate_convex_polygon_with_holes(&outer, &[&first, &second])
            .expect("subdivided rectangular panel with circular holes triangulates");
    }

    #[test]
    fn concave_l_shape_is_triangulated_and_refined() {
        let polygon = [
            (0.0, 0.0),
            (2.0, 0.0),
            (2.0, 1.0),
            (1.0, 1.0),
            (1.0, 2.0),
            (0.0, 2.0),
        ];
        let triangles = triangulate_simple_polygon(&polygon).unwrap();
        assert_eq!(triangles.len(), polygon.len() - 2);
        let mesh = polygon_mesh_triangles(&polygon, 2, 7).unwrap();
        assert_eq!(mesh.elements.len(), triangles.len() * 16);
        assert!(mesh.boundaries.iter().all(|boundary| boundary.marker == 7));
        let area: f64 = mesh
            .elements
            .iter()
            .map(|element| {
                let a = mesh.nodes[element.nodes[0]];
                let b = mesh.nodes[element.nodes[1]];
                let c = mesh.nodes[element.nodes[2]];
                0.5 * ((b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x)).abs()
            })
            .sum();
        assert!((area - 3.0).abs() < 1.0e-12);
    }
}
