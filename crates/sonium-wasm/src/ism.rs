//! Image-source geometry helpers used by the room solver.

use super::{LShapedRoom, Point3D};

pub(super) fn configured_reflection_order(method: &str) -> usize {
    match method {
        "direct" | "modal" | "bem" => 0,
        "image-source-1" => 1,
        "image-source-2" => 2,
        "image-source-3" => 3,
        "hybrid" | "hybrid-bem" => 2,
        _ => 2,
    }
}

pub(super) fn l_shaped_first_order_images(
    room: &LShapedRoom,
    source: &Point3D,
) -> Vec<(Point3D, &'static str)> {
    let total_depth = room.depth1 + room.depth2;
    let mut images = Vec::new();

    images.push((Point3D::new(-source.x, source.y, source.z), "left"));

    if source.y <= room.depth1 {
        images.push((
            Point3D::new(2.0 * room.width1 - source.x, source.y, source.z),
            "right",
        ));
    } else {
        images.push((
            Point3D::new(2.0 * room.width2 - source.x, source.y, source.z),
            "right",
        ));
    }

    images.push((Point3D::new(source.x, -source.y, source.z), "front"));
    images.push((
        Point3D::new(source.x, 2.0 * total_depth - source.y, source.z),
        "back",
    ));
    images.push((Point3D::new(source.x, source.y, -source.z), "floor"));
    images.push((
        Point3D::new(source.x, source.y, 2.0 * room.height - source.z),
        "ceiling",
    ));

    if source.y < room.depth1 && source.x > room.width2 {
        images.push((
            Point3D::new(source.x, 2.0 * room.depth1 - source.y, source.z),
            "step_horizontal",
        ));
        images.push((
            Point3D::new(2.0 * room.width2 - source.x, source.y, source.z),
            "step_vertical",
        ));
    }

    if source.y > room.depth1 && source.x < room.width2 {
        images.push((
            Point3D::new(2.0 * room.width2 - source.x, source.y, source.z),
            "step_vertical",
        ));
    }

    images
}

pub(super) fn is_valid_l_shaped_image_source(
    room: &LShapedRoom,
    image: &Point3D,
    listener: &Point3D,
) -> bool {
    let segments = 10;
    for i in 1..segments {
        let t = i as f64 / segments as f64;
        let test_point = Point3D::new(
            listener.x + t * (image.x - listener.x),
            listener.y + t * (image.y - listener.y),
            listener.z + t * (image.z - listener.z),
        );

        if !room.contains(&test_point) && i < segments - 1 {
            let prev_t = (i - 1) as f64 / segments as f64;
            let prev_point = Point3D::new(
                listener.x + prev_t * (image.x - listener.x),
                listener.y + prev_t * (image.y - listener.y),
                listener.z + prev_t * (image.z - listener.z),
            );
            if !room.contains(&prev_point) {
                // The path was already outside and may pass through a corner.
                // Retain the historical heuristic and let the forbidden-region
                // check below make the final decision.
            }
        }
    }

    let total_depth = room.depth1 + room.depth2;
    !(image.x > room.width2 && image.y > room.depth1 && image.y < total_depth)
}
