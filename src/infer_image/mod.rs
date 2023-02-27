use std::convert::TryInto;
use wasi_nn;

pub fn graph_inference(weights : &[u8], tensor_data : &[u8], input_dims : &[u32], output_index : u32, output_dims : usize )-> Result< Vec<f32>, Box<dyn std::error::Error> >{
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
    let tensor = wasi_nn::Tensor {
        dimensions: &input_dims,
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
    let mut output_buffer = vec![0f32; output_dims];
    // println!("output_buffer: {:?}", output_buffer);
    unsafe {
       wasi_nn::get_output(
           context,
           output_index,
           &mut output_buffer[..] as *mut [f32] as *mut u8,
           (output_buffer.len() * 4).try_into().unwrap(),
       )
       .unwrap()
   };

   let output_buffer: Vec<f32> = output_buffer
            .iter()
            .cloned()
            .collect();

   return Ok(output_buffer)
}

