use image::{Rgb, RgbImage};
use imageproc::{
    drawing::{draw_hollow_rect_mut, draw_text, draw_cross_mut},
    rect::Rect,
};
use lazy_static::lazy_static;

lazy_static! {
    static ref DEJAVU_MONO: rusttype::Font<'static> = {
        let font_data: &[u8] = include_bytes!("../../resources/DejaVuSansMono.ttf");
        let font: rusttype::Font<'static> =
            rusttype::Font::try_from_bytes(font_data).expect("Load font");
        font
    };
}

pub fn draw_bboxes_on_image(
    mut frame_rgb: RgbImage,
    bbox_coord : Vec<f32>, // later give box along with its confidence
    confidence : f32,
) -> RgbImage{ 
    let (blue , _red)  = (Rgb([0u8,   0u8,   255u8]), Rgb([255u8,   0u8,   0u8]));

    let face_rect =
                Rect::at(bbox_coord[0] as i32, bbox_coord[1] as i32).of_size((bbox_coord[2] - bbox_coord[0]) as u32, (bbox_coord[3] - bbox_coord[1])  as u32);
    draw_hollow_rect_mut(&mut frame_rgb, face_rect, blue);
    frame_rgb = draw_text(
                    &frame_rgb,
                    blue,
                    bbox_coord[0] as i32,
                    bbox_coord[1] as i32,
                    rusttype::Scale { x: 16.0, y: 16.0 },
                    &DEJAVU_MONO,
                    &format!("{:.2}%", confidence * 100.0),
                );
    for i in 0.. 6{
        draw_cross_mut(&mut frame_rgb, blue, bbox_coord[4+(2*i)] as i32, bbox_coord[5+(2*i)] as i32);
    }
    // draw_hollow_rect_mut(&mut image_dr, Rect::at(60, 10).of_size(20, 20), blue);
    return frame_rgb;
}