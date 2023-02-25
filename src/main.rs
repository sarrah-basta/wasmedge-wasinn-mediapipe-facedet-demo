use image::io::Reader;
use image::{Rgb, RgbImage, DynamicImage};
use std::convert::TryInto;
use std::env;
use std::fs;
use std::path::Path;
use wasi_nn;
use lazy_static::lazy_static;

use imageproc::{
    drawing::{draw_hollow_rect_mut, draw_text, draw_cross_mut},
    rect::Rect,
};

/// Positive additive constant to avoid divide-by-zero.
const EPS: f32 = 1.0e-7;

pub fn main() -> Result<(), Box<dyn std::error::Error>> {
    main_entry()?;
    Ok(())
}

#[no_mangle]
fn main_entry() -> Result<(), Box<dyn std::error::Error>> {
    infer_image()?;
    Ok(())
}

fn infer_image()-> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();
    let model_bin_name: &str = &args[1];
    let image_name: &str = &args[2];

    let weights = fs::read(model_bin_name).unwrap();
    println!("Read graph weights, size in bytes: {}", weights.len());

    let pixels = Reader::open(image_name.to_string()).unwrap().decode().unwrap();
    let inp_img: DynamicImage = pixels;
    let image_height : f32 = inp_img.height() as f32;
    let image_width : f32 = inp_img.width() as f32;


    let graph = unsafe {
        wasi_nn::load(
            &[&weights],
            4, // encoding for tflite: wasi_nn::GRAPH_ENCODING_TENSORFLOWLITE
            wasi_nn::EXECUTION_TARGET_CPU,
        )
        .unwrap()
    };
    println!("Loaded graph into wasi-nn with ID: {}", graph);

    let context = unsafe { wasi_nn::init_execution_context(graph).unwrap() };
    println!("Created wasi-nn execution context with ID: {}", context);

    // Load a tensor that precisely matches the graph input tensor (see
    let tensor_data = image_to_tensor(image_name.to_string(), 128, 128);
    println!("Read input tensor, size in bytes: {:?}", tensor_data.len());

    let tensor = wasi_nn::Tensor {
        dimensions: &[1, 128, 128, 3],
        r#type: wasi_nn::TENSOR_TYPE_F32,
        data: &tensor_data,
    };
    unsafe {
        wasi_nn::set_input(context, 0, tensor).unwrap();
    }
    // Execute the inference.
    unsafe {
        wasi_nn::compute(context).unwrap();
    }
    println!("Executed graph inference");
    // Retrieve the output.
    let mut output_buffer = vec![0f32; 896 * 16];
    // println!("output_buffer: {:?}", output_buffer);
    unsafe {
       wasi_nn::get_output(
           context,
           0,
           &mut output_buffer[..] as *mut [f32] as *mut u8,
           (output_buffer.len() * 4).try_into().unwrap(),
       )
       .unwrap()
   };
   // Original output0 obtained as anchors
    let output_0: Vec<f32> = output_buffer
            .iter()
            .cloned()
            .collect();
    let mut output0: Vec<Vec<f32>> = output_0.chunks(16).map(|x| x.try_into().unwrap()).collect();

   // Bounding box redefined as `[x_top_left, y_top_left, x_bottom_right, y_bottom_right, x_Left eye, y_Left eye, x_Right eye ...]
    conv_anchors(&mut output0, image_width, image_height);
    let mut output1 = vec![0f32; 896 * 1];
    // println!("output_buffer: {:?}", output_buffer);
    unsafe {
       wasi_nn::get_output(
           context,
           1,
           &mut output1[..] as *mut [f32] as *mut u8,
           (output1.len() * 4).try_into().unwrap(),
       )
       .unwrap()
    };

    
    // in the tflite model, the scores are not returned as probabilities but as sigmoid scores
    output1 = output1.iter().map(|x| 1.0/(1.0 + f32::exp(-(x)))).collect();
    // Fuse bounding boxes with confidence scores (converted to probabilities)
    // Filter out bounding boxes with a confidence score below the threshold
    let mut bboxes_with_confidences: Vec<_> = output0
        .iter()
        .zip(output1.iter())
        .filter_map(|(bbox, confidence)| match confidence {
            x if *x > 0.7 => Some((bbox,confidence)),
            _ => None,
        })
        .collect();

    // Sort pairs of bounding boxes with confidence scores by **ascending** confidences to allow
    // cheap removal of the top candidates from the back
    bboxes_with_confidences.sort_by(|a, b| a.1.partial_cmp(b.1).unwrap());

    // println!("\n\nboxes before nms : {:?}\n", bboxes_with_confidences);
    // Run non-maximum suppression on the sorted vector of bounding boxes with confidences
    let selected_bboxes = non_maximum_suppression(bboxes_with_confidences, 0.7);
    // println!("\n\nboxes after nms : {:?}\n", selected_bboxes);

    let mut frame_rgb: RgbImage = inp_img.to_rgb8();
    let out_name = format!("{}{}",image_name,"_drawn_out.jpg");
    println!("\nThe number of distinct faces detected (AFTER NMS) are : {:?} \n", selected_bboxes.len());
    for item in selected_bboxes.iter(){
        println!("\nThe pixel co-ordinates for the bounding box of the detected faces with probability {:?} are : \n", item.1);
        println!("(x1, y1) : {:?} , {:?}", item.0[0] as u32, item.0[1] as u32 );
        println!("(x2, y2) : {:?} , {:?}", item.0[2] as u32, item.0[3] as u32 );

        println!("The pixel co-ordinates of the facial keypoints are :");
        println!("Left eye : {:?} , {:?} ", item.0[4] as u32, item.0[5] as u32);
        println!("Right eye : {:?} , {:?}", item.0[6] as u32, item.0[7] as u32);
        println!("Nose Tip : {:?} , {:?} ", item.0[8] as u32, item.0[9]as u32);
        println!("Mouth : {:?} , {:?} ", item.0[10] as u32, item.0[11] as u32);
        println!("Left eye tragion : {:?} , {:?} ", item.0[12] as u32, item.0[13] as u32);
        println!("Right eye tragion : {:?} , {:?} ",  item.0[14] as u32, item.0[15] as u32);
        frame_rgb = draw_bboxes_on_image(frame_rgb, item.0.clone() , item.1);
    }
    println!("\n The output image is saved in the same place as {:?}", &image_name);

    let path = Path::new(&out_name);
    frame_rgb.save(path).unwrap();

   Ok(output_buffer)
}

fn gen_anchors() -> Vec<Vec<f32>> {
    let mut anchors = vec![];
    let input_size_width = 128;
    let input_size_height = 128;
    let min_scale = 0.1484375;
    let max_scale = 0.75;
    let anchor_offset_x = 0.5;
    let anchor_offset_y = 0.5;
    let num_layers = 4;
    let feature_map_width_og = vec![];
    let feature_map_height_og = vec![];
    let strides = vec![8, 16, 16, 16];
    let aspect_ratios_og = vec![1.0];
    let reduce_boxes_in_lowest_layer = false;
    let interpolated_scale_aspect_ratio = 1.0;
    let fixed_anchor_size = true;
    let feature_map_width_size = feature_map_width_og.len();
    let feature_map_height_size = feature_map_height_og.len();
    let strides_size = strides.len();
    let aspect_ratios_size = aspect_ratios_og.len();
    if strides_size != num_layers {
        println!("{:?} ","strides_size and num_layers must be equal.");
        return vec![];
    }
    let mut layer_id = 0;
    while layer_id < strides_size {
        let mut anchor_height = vec![];
        let mut anchor_width = vec![];
        let mut aspect_ratios = vec![];
        let mut scales = vec![];

        //for same strides, merge anchors in same order
        let mut last_same_stride_layer = layer_id;
        while last_same_stride_layer < strides_size && strides[last_same_stride_layer] == strides[layer_id] {
            let scale = min_scale + ((((max_scale - min_scale)*1.0)*last_same_stride_layer as f32)/(strides_size as f32 - 1.0));
            if last_same_stride_layer == 0 && reduce_boxes_in_lowest_layer {
                aspect_ratios.push(1.0);
                aspect_ratios.push(2.0);
                aspect_ratios.push(0.5);
                scales.push(0.1);
                scales.push(scale);
                scales.push(scale);
            } else {
                for aspect_ratio_id in 0..aspect_ratios_size {
                    aspect_ratios.push(aspect_ratios_og[aspect_ratio_id]);
                    scales.push(scale);
                }
                if interpolated_scale_aspect_ratio > 0.0 {
                    let scale_next = if last_same_stride_layer == (strides_size - 1) { 1.0 } else { min_scale + ((((max_scale - min_scale)*1.0)*(last_same_stride_layer + 1) as f32)/(strides_size as f32 - 1.0)) };
                    scales.push(f32::sqrt(scale*scale_next));
                    aspect_ratios.push(interpolated_scale_aspect_ratio);
                }
            }
            last_same_stride_layer += 1;
        }
        for i in 0..aspect_ratios.len() {
            let ratio_sqrts = f32::sqrt(aspect_ratios[i]);
            anchor_height.push(scales[i]/ratio_sqrts);
            anchor_width.push(scales[i]*ratio_sqrts);
        }
        let feature_map_height;
        let feature_map_width;
        if feature_map_height_size > 0 || feature_map_width_size  > 0{
            feature_map_height = feature_map_height_og[layer_id];
            feature_map_width = feature_map_width_og[layer_id];
        } else {
            let stride = strides[layer_id];
            feature_map_height = (1* input_size_height)/stride;
            feature_map_width = (1* input_size_width)/stride;
        }
        for y in 0..feature_map_height {
            for x in 0..feature_map_width {
                for anchor_id in 0..anchor_height.len() {
                    let x_center = (x as f32 + anchor_offset_x)*1.0/feature_map_width as f32;
                    let y_center = (y as f32 + anchor_offset_y)*1.0/feature_map_height as f32;
                    let w;
                    let h;
                    if fixed_anchor_size {
                        w = 1.0;
                        h = 1.0;
                    } else {
                        w = anchor_width[anchor_id];
                        h = anchor_height[anchor_id];
                    }
                    let new_anchor = vec![x_center, y_center, h, w];
                    anchors.push(new_anchor);
                }
            }
        }
        layer_id = last_same_stride_layer;
    }
    return anchors;
}

fn conv_anchors(output0 : &mut Vec<Vec<f32>>, image_width : f32, image_height : f32){

    let anchors = gen_anchors();
    let (input_width, input_height) = (128, 128);
    let iter = output0.iter_mut().zip(anchors.iter());
    for (out,anc) in iter{
        // println!(" \n\n this time \n");
        let mut out_iter = out.iter();
        let mut out_ans : Vec<f32> = vec![];
        let cx = (out_iter.next().unwrap() + (anc[0] * input_width as f32)) / input_width as f32;
        let cy = (out_iter.next().unwrap() + (anc[1] * input_width as f32)) / input_height as f32;
        let w = (out_iter.next().unwrap()) / input_width as f32;
        let h = (out_iter.next().unwrap()) / input_height as f32;
        out_ans.extend_from_slice(&vec![((cx - (w*0.5)) * image_width ), ((cy - (h*0.5)) * image_height ), ((cx + (w*0.5)) * image_width ), ((cy + (h*0.5)) * image_height )]);

        let keypoints : Vec<f32> = out_iter.enumerate()
                            .map(|(index, x)| {
                                if index % 2 == 0 {
                                    ((x+anc[0]*input_width as f32)/input_width as f32) * image_width
                                } else {
                                    ((x+anc[1]*input_height as f32)/input_height as f32) * image_height
                                }
                            })
                            .collect();
        out_ans.extend_from_slice(&keypoints);
        *out = out_ans;
    }
} 

// Take the image located at 'path', open it, resize it to height x width, and then converts
// the pixel precision to FP32 by normalizing between -1 to 1.
// The resulting *RGB* pixel vector is then returned.
fn image_to_tensor(path: String, height: u32, width: u32) -> Vec<u8> {
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
    //   let u8_f32: f32 = -0.5686275_f32;
      let u8_bytes = u8_f32.to_ne_bytes();
      for j in 0..4 {
        u8_f32_arr[(i * 4) + j] = u8_bytes[j];
      }
    }
    return u8_f32_arr;
  }

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
fn non_maximum_suppression(
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

/// Calculate the intersection-over-union metric for two bounding boxes.
fn iou(bbox_a: &Vec<f32>, bbox_b: &Vec<f32>) -> f32 {
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
fn bbox_area(bbox: &Vec<f32>) -> f32 {
    let height = bbox[3] - bbox[1];
    let width = bbox[2] - bbox[0];
    if width < 0.0 || height < 0.0 {
        // bbox is empty/undefined since the bottom-right corner is above the top left corner
        return 0.0;
    }

    width * height
}

lazy_static! {
    static ref DEJAVU_MONO: rusttype::Font<'static> = {
        let font_data: &[u8] = include_bytes!("../resources/DejaVuSansMono.ttf");
        let font: rusttype::Font<'static> =
            rusttype::Font::try_from_bytes(font_data).expect("Load font");
        font
    };
}

fn draw_bboxes_on_image(
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

// A wrapper for class ID and match probabilities.
#[derive(Debug, PartialEq)]
struct InferenceResult(usize, u8);
