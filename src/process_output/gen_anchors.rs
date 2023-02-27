pub fn gen_anchors() -> Vec<Vec<f32>> {
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