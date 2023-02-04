use image::io::Reader;
use image::DynamicImage;
use image::GenericImageView;
use std::convert::TryInto;
use std::env;
use std::fs;
use fastrand;
use wasi_nn;

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
   let mut output0 : Vec<Vec<f32>> = vec![];
   for i in 0..896{ 
        let mut ans_i : Vec<f32> = vec![];
        for j in 0..16 {
            ans_i.push(output_buffer[i*16 + j]);
        }
        output0.push(ans_i);
    }
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
    let answer = filter_detections(output1);
    let scores = answer.0;
    let detection = answer.1;
    // println!("scores.shape{:?}",scores);
    
    let detections = extract_detections(output0, detection, scores);
    let boxx = detections.0;
    let keypoints = detections.1 ;
    let prob_value = detections.2 ;
    // println!("box : {:?} \n keypoints : {:?} \n\n", boxx, keypoints);

    let mut facial_key : Vec<u32> = vec![];
    //get pixel co-ordinates for bounding box
    let x1 = (image_width * boxx[0] ) as u32;
    let x2 = (image_width * boxx[2] ) as u32;
    let y1 = (image_height * boxx[1] ) as u32;
    let y2 = (image_height * boxx[3] ) as u32;
    println!("\n The pixel co-ordinates for the bounding box of the detected face with probability {:?} are : \n", prob_value);
    println!("(x1, y1) : {:?} , {:?}", x1, y1);
    println!("(x2, y2) : {:?} , {:?}", x2, y2);

    facial_key.push(x1);
    facial_key.push(y1);
    facial_key.push(x2);
    facial_key.push(y2);
    
    for i in 0..6{
        let x_keypoint = (image_width * keypoints[i][0]) as u32;
        let y_keypoint = (image_height * keypoints[i][1]) as u32;
        facial_key.push(x_keypoint);
        facial_key.push(y_keypoint);
    }
    println!("\n The pixel co-ordinates of the facial keypoints are : \n");
    println!("Left eye : {:?} , {:?} ", facial_key[4], facial_key[5]);
    println!("Right eye : {:?} , {:?}", facial_key[6], facial_key[7]);
    println!("Nose Tip : {:?} , {:?} ", facial_key[8], facial_key[9]);
    println!("Mouth : {:?} , {:?} ", facial_key[10], facial_key[11]);
    println!("Left eye tragion : {:?} , {:?} ", facial_key[12], facial_key[13]);
    println!("Right eye tragion : {:?} , {:?} ", facial_key[14], facial_key[15]);

    println!("\n Writing output Vector facial_key to file output_facial_key.bin ");
    let output_facial_key: Vec<u8> = facial_key.iter().flat_map(|val| val.to_be_bytes()).collect();  
    std::fs::write("output_facial_key.bin", output_facial_key).unwrap(); 
    
    
   Ok(output_buffer)
}

fn filter_detections(output1: Vec<f32>) -> (Vec<f32>,Vec<u32>) {
    let sigmoid_score_threshold = f32::ln(0.7/(1.0 - 0.7));
    // println!("{:?} {:?} ","sigmoid score threshold ", sigmoid_score_threshold);
    let mut good_detections = vec![];
    let mut scores = vec![];
    for (i, value) in output1.iter().enumerate() {
        if value > &sigmoid_score_threshold {
            good_detections.push(i as u32);
            let score_i = 1.0/(1.0 + f32::exp(-(value)));
            scores.push(score_i);
        }
    }
    // println!("good_detections1.shape :{:?}", good_detections);
    return (scores, good_detections);
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

// pass output of information about the detected faces along with indices with probability scores over the threshold
fn extract_detections(output0: Vec<Vec<f32>>, good_detection_indices : Vec<u32>, score_probs : Vec<f32>) -> (Vec<f32>, Vec<Vec<f32>>, f32) {

    // picking a random box to draw among good_detection_indices
    // later NMS should be applied here
    let rng = fastrand::usize(..good_detection_indices.len());
    let detection_idx = good_detection_indices[rng] as usize;
    let score_idx = score_probs[rng];

    let (input_width, input_height) = (128, 128);
    let anchors : Vec<Vec<f32>> = gen_anchors();
    let anchor = &anchors[detection_idx];
    let sx = output0[detection_idx][0];
    let sy = output0[detection_idx][1];
    let mut w = output0[detection_idx][2];
    let mut h = output0[detection_idx][3];
    let mut cx = sx + (anchor[0] * input_width as f32);
    let mut cy = sy + (anchor[1] * input_height as f32);
    cx /= input_width as f32;
    cy /= input_height as f32;
    w /= input_width as f32;
    h /= input_height as f32;
    let mut keypoints_idx = vec![];
    for j in 0..6 {
        let mut lx = output0[detection_idx][((4 + (2*j)) + 0)];
        let mut ly = output0[detection_idx][((4 + (2*j)) + 1)];
        lx += anchor[0] * input_width as f32;
        ly += anchor[1] * input_height as f32;
        lx /= input_width as f32;
        ly /= input_height as f32;
        let mut keypoints_idx_j = vec![];
        keypoints_idx_j.push(lx);
        keypoints_idx_j.push(ly);
        keypoints_idx.push(keypoints_idx_j);
    }
    let mut boxes_idx = vec![];
    boxes_idx.push(cx - (w*0.5));
    boxes_idx.push(cy - (h*0.5));
    boxes_idx.push(cx + (w*0.5));
    boxes_idx.push(cy + (h*0.5));
    return (boxes_idx, keypoints_idx, score_idx);
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

    // println!("{:?}", u8_f32_arr); 
    return u8_f32_arr;
  }

// A wrapper for class ID and match probabilities.
#[derive(Debug, PartialEq)]
struct InferenceResult(usize, u8);
