// use std::{
//     collections::HashMap,
//     sync::Arc,
//     time::{Duration, Instant},
// };

// use image::{Rgb, RgbImage};
// use imageproc::{
//     drawing::{draw_hollow_rect, draw_text},
//     rect::Rect,
// };

// /// Draw bounding boxes with confidence scores on the image.
// fn draw_bboxes_on_image(
//     mut frame: RgbImage,
//     bboxes_with_confidences: Vec<([f32; 4], f32)>,
//     width: u32,
//     height: u32,
// ) -> RgbImage {
//     let (width, height) = (width as f32, height as f32);

//     let color = Rgb::from([0, 255, 0]);

//     for (bbox, confidence) in bboxes_with_confidences.iter() {
//         // Coordinates of top-left and bottom-right points
//         // Coordinate frame basis is on the top left corner
//         let (x_tl, y_tl) = (bbox[0] * width, bbox[1] * height);
//         let (x_br, y_br) = (bbox[2] * width, bbox[3] * height);
//         let rect_width = x_br - x_tl;
//         let rect_height = y_br - y_tl;

//         let face_rect =
//             Rect::at(x_tl as i32, y_tl as i32).of_size(rect_width as u32, rect_height as u32);

//         frame = draw_hollow_rect(&frame, face_rect, Rgb::from([0, 255, 0]));
//         frame = draw_text(
//             &frame,
//             color,
//             x_tl as i32,
//             y_tl as i32,
//             rusttype::Scale { x: 16.0, y: 16.0 },
//             &DEJAVU_MONO,
//             &format!("{:.2}%", confidence * 100.0),
//         );
//     }

//     frame
// }