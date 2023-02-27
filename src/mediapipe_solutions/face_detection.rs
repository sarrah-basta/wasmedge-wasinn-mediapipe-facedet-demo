use std::fs;
use std::convert::TryInto;
use std::path::Path;
use image::{RgbImage, DynamicImage};
use image::io::Reader;

use crate::process_image::image_to_tensor;
use crate::process_output::conv_anchors::*;
use crate::process_output::non_maximum_suppression::*;
use crate::show_output::draw_output::*;
use crate::infer_image::graph_inference;
use crate::show_output::print_output::*;

pub fn face_detection_short(model_bin_name: &str,image_name: &str)-> Result<(), Box<dyn std::error::Error>> {

    let weights = fs::read(model_bin_name).unwrap();
    println!("Read graph weights, size in bytes: {}", weights.len());

    let pixels = Reader::open(image_name.to_string()).unwrap().decode().unwrap();
    let inp_img: DynamicImage = pixels;
    let image_height : f32 = inp_img.height() as f32;
    let image_width : f32 = inp_img.width() as f32;

    // infer_image(&weights, &tensor_data, output_index, output_dims);
//     let graph = unsafe {
//         wasi_nn::load(
//             &[&weights],
//             4, // encoding for tflite: wasi_nn::GRAPH_ENCODING_TENSORFLOWLITE
//             wasi_nn::EXECUTION_TARGET_CPU,
//         )
//         .unwrap()
//     };
//     println!("Loaded graph into wasi-nn with ID: {}", graph);

//     let context = unsafe { wasi_nn::init_execution_context(graph).unwrap() };
//     println!("Created wasi-nn execution context with ID: {}", context);

//     // Load a tensor that precisely matches the graph input tensor (see
//     let tensor_data = image_to_tensor(image_name.to_string(), 128, 128);
//     println!("Read input tensor, size in bytes: {:?}", tensor_data.len());

//     let tensor = wasi_nn::Tensor {
//         dimensions: &[1, 128, 128, 3],
//         r#type: wasi_nn::TENSOR_TYPE_F32,
//         data: &tensor_data,
//     };
//     unsafe {
//         wasi_nn::set_input(context, 0, tensor).unwrap();
//     }
//     // Execute the inference.
//     unsafe {
//         wasi_nn::compute(context).unwrap();
//     }
//     println!("Executed graph inference");
//     // Retrieve the output.
//     let mut output_buffer = vec![0f32; 896 * 16];
//     // println!("output_buffer: {:?}", output_buffer);
//     unsafe {
//        wasi_nn::get_output(
//            context,
//            0,
//            &mut output_buffer[..] as *mut [f32] as *mut u8,
//            (output_buffer.len() * 4).try_into().unwrap(),
//        )
//        .unwrap()
//    };

   // Load a tensor that precisely matches the graph input tensor (see
    let tensor_data = image_to_tensor(image_name.to_string(), 128, 128);
    println!("Read input tensor, size in bytes: {:?}", tensor_data.len());

    // let tensor = wasi_nn::Tensor {
    //     dimensions: &[1, 128, 128, 3],
    //     r#type: wasi_nn::TENSOR_TYPE_F32,
    //     data: &tensor_data,
    // };

    // Original output0 obtained as anchors
    let output_0 : Vec<f32> = graph_inference(&weights, &tensor_data, &[1, 128, 128, 3], 0, 896*16).unwrap();
    let output_1 : Vec<f32> = graph_inference(&weights, &tensor_data, &[1, 128, 128, 3], 1, 896*1).unwrap(); 

    // collecting output0 to make it a 2d vector of 896*16
    let mut output0: Vec<Vec<f32>> = output_0.chunks(16).map(|x| x.try_into().unwrap()).collect();

   // Bounding box redefined as `[x_top_left, y_top_left, x_bottom_right, y_bottom_right, x_Left eye, y_Left eye, x_Right eye ...]
    conv_anchors(&mut output0, image_width, image_height);
    // let mut output1 = vec![0f32; 896 * 1];
    // // println!("output_buffer: {:?}", output_buffer);
    // unsafe {
    //    wasi_nn::get_output(
    //        context,
    //        1,
    //        &mut output1[..] as *mut [f32] as *mut u8,
    //        (output1.len() * 4).try_into().unwrap(),
    //    )
    //    .unwrap()
    // };

    
    // in the tflite model, the scores are not returned as probabilities but as sigmoid scores
    let output1 : Vec<f32> = output_1.iter().map(|x| 1.0/(1.0 + f32::exp(-(x)))).collect();
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
        printf_face_detection_short(&item);
        frame_rgb = draw_bboxes_on_image(frame_rgb, item.0.clone() , item.1);
    }
    let path = Path::new(&out_name);
    frame_rgb.save(path).unwrap();

    println!("\n The output image is saved at {:?}", &path);
    Ok(())
}