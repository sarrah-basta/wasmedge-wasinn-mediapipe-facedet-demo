/// Run non-maximum-suppression on candidate bounding boxes.
///
/// The pairs of bounding boxes with confidences have to be sorted in **ascending** order of
/// confidence because we want to `pop()` the most confident elements from the back.
///
/// Start with the most confident bounding box and iterate over all other bounding boxes in the
/// order of decreasing confidence. Grow the vector of selected bounding boxes by adding only those
/// candidates which do not have a IoU scores above `max_iou` with already chosen bounding boxes.
/// This iterates over all bounding boxes in `sorted_bboxes_with_confidences`. Any candidates with
/// scores generally too low to be considered should be filtered out before.
pub fn non_maximum_suppression(
    mut sorted_bboxes_with_confidences: Vec<(&Vec<f32>, &f32)>,
    max_iou: f32,
) -> Vec<(Vec<f32>, f32)> {
    let mut selected = vec![];
    'candidates: loop {
        // Get next most confident bbox from the back of ascending-sorted vector.
        // All boxes fulfill the minimum confidence criterium.
        match sorted_bboxes_with_confidences.pop() {
            Some((bbox, confidence)) => {
                // Check for overlap with any of the selected bboxes
                for (selected_bbox, _) in selected.iter() {
                    match iou(bbox, selected_bbox) {
                        x if x > max_iou => continue 'candidates,
                        _ => (),
                    }
                }

                // bbox has no large overlap with any of the selected ones, add it
                selected.push(((*bbox.clone()).to_vec(), *confidence))
            }
            None => break 'candidates,
        }
    }

    selected
}

// Positive additive constant to avoid divide-by-zero.
const EPS: f32 = 1.0e-7;

// Calculate the intersection-over-union metric for two bounding boxes.
pub fn iou(bbox_a: &Vec<f32>, bbox_b: &Vec<f32>) -> f32 {
    // Calculate corner points of overlap box
    // If the boxes do not overlap, the corner-points will be ill defined, i.e. the top left
    // corner point will be below and to the right of the bottom right corner point. In this case,
    // the area will be zero.
    let overlap_box: Vec<f32> = vec![
        f32::max(bbox_a[0], bbox_b[0]),
        f32::max(bbox_a[1], bbox_b[1]),
        f32::min(bbox_a[2], bbox_b[2]),
        f32::min(bbox_a[3], bbox_b[3]),
    ];

    let overlap_area = bbox_area(&overlap_box);

    // Avoid division-by-zero with `EPS`
    overlap_area / (bbox_area(bbox_a) + bbox_area(bbox_b) - overlap_area + EPS)
}

/// Calculate the area enclosed by a bounding box.
///
/// The bounding box is passed as four-element array defining 8 distinct points:
/// `[x_top_left, y_top_left, x_bottom_right, y_bottom_right]`
/// If the bounding box is ill-defined by having the bottom-right point above/to the left of the
/// top-left point, the area is zero.
pub fn bbox_area(bbox: &Vec<f32>) -> f32 {
    let height = bbox[3] - bbox[1];
    let width = bbox[2] - bbox[0];
    if width < 0.0 || height < 0.0 {
        // bbox is empty/undefined since the bottom-right corner is above the top left corner
        return 0.0;
    }

    width * height
}