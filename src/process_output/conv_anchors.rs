use crate::process_output::gen_anchors::*;

pub fn conv_anchors(output0 : &mut Vec<Vec<f32>>, image_width : f32, image_height : f32){

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