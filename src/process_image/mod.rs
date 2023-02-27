use image::io::Reader;
use image::{DynamicImage};

// Take the image located at 'path', open it, resize it to height x width, and then converts
// the pixel precision to FP32 by normalizing between -1 to 1.
// The resulting *RGB* pixel vector is then returned.
pub fn image_to_tensor(path: String, height: u32, width: u32) -> Vec<u8> {
    let pixels = Reader::open(path).unwrap().decode().unwrap();
    let dyn_img: DynamicImage = pixels.resize_exact(width, height, image::imageops::Nearest);
    let bgr_img = dyn_img.to_rgb8();

    // Get an array of the pixel values
    let raw_u8_arr: &[u8] = &bgr_img.as_raw()[..];
    let norm = raw_u8_arr.to_vec().iter().map( |&e| e as f32 / 255.0 ).collect::<Vec<f32>>();  
    let full_norm = norm.iter().map( |&e| e as f32 - 0.5  ).collect::<Vec<f32>>(); 
    let full2_norm = full_norm.iter().map( |&e| e as f32 / 0.5).collect::<Vec<f32>>(); 

    // Create an array to hold the f32 value of those pixels
    let bytes_required = full2_norm.len() * 4;

    // let bytes_required = raw_u8_arr.len() * 4;
    let mut u8_f32_arr: Vec<u8> = vec![0; bytes_required];

    for i in 0..full2_norm.len() {
      // Read the number as a f32 and break it into u8 bytes
      let u8_f32: f32 = full2_norm[i];
      let u8_bytes = u8_f32.to_ne_bytes();
      for j in 0..4 {
        u8_f32_arr[(i * 4) + j] = u8_bytes[j];
      }
    }
    return u8_f32_arr;
  }
