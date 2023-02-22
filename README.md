# MediaPipe Face Detection Example For WASI-NN with Tensorflow Lite Backend

This package is a high-level Rust bindings for [wasi-nn] example of BlazeFace(https://arxiv.org/abs/1907.05047) with Tensorflow Lite backend.

[wasi-nn]: https://github.com/WebAssembly/wasi-nn

## About the Model
MediaPipe Face Detection is an ultrafast face detection solution that comes with 6 landmarks and multi-face support. It is based on [BlazeFace](https://arxiv.org/abs/1907.05047), a lightweight and well-performing face detector tailored for mobile GPU inference.

The details of the tf-lite model used can be found in this [MediaPipe Model Card](https://drive.google.com/file/d/1d4-xJP9PVzOvMBDgIjz6NhvpnlG9_i0S/preview).

## Dependencies

This crate depends on the `wasi-nn` in the `Cargo.toml`:

```toml
[dependencies]
wasi-nn = "0.1.0"
image  = "0.23.14"
fastrand = "1.8.0" 
```

> Note: After the `TENSORFLOWLITE` encoding added into the wasi-nn crate, we'll update this example to use the newer version.

## Build

Compile the source code to WebAssembly:

```bash
cargo build --target=wasm32-wasi --release
```

The output WASM file will be at [`target/wasm32-wasi/release/wasmedge-wasinn-example-tflite-mediapipe-face-detection`](target/wasm32-wasi/release/wasmedge-wasinn-example-tflite-mediapipe-face-detection.wasm).
To speed up the image processing, we can enable the AOT mode in WasmEdge with:

```bash
wasmedgec target/wasm32-wasi/release/wasmedge-wasinn-example-tflite-mediapipe-face-detection.wasm out.wasm
```

## Run

### Required Files

The testing images are located at `./images`:

- ./images/test_image.jpg

![6 faces in detection](./images/test_image.jpg)
- ./images/garrett-jackson-auTAb39ImXg-unsplash.jpg
- ./images/radu-florin-JyVcAIUAcPM-unsplash.jpg


The `tflite` model is located at `./face_detection_short_range.tflite`

### Output

Users should [install the WasmEdge with WASI-NN TensorFlow-Lite backend plug-in](https://wasmedge.org/book/en/write_wasm/rust/wasinn.html#get-wasmedge-with-wasi-nn-plug-in-tensorflow-lite-backend).

Execute the WASM with the `wasmedge` with Tensorflow Lite supporting:

```bash
wasmedge --dir .:. out.wasm face_detection_short_range.tflite /path/to/image_file.jpg/
```

This selects a *random* box from among the detections (filtered using confidence score threshold) and gives the output

```bash
wasmedge --dir .:. out.wasm face_detection_short_range.tflite images/test_image.jpg 
```
```console
Read graph weights, size in bytes: 229032
Loaded graph into wasi-nn with ID: 0
Created wasi-nn execution context with ID: 0
Read input tensor, size in bytes: 196608
Executed graph inference

 The pixel co-ordinates for the bounding box of the detected face with probability 0.81526 are : 

(x1, y1) : 359 , 277
(x2, y2) : 518 , 408

 The pixel co-ordinates of the facial keypoints are : 

Left eye : 399 , 312 
Right eye : 459 , 315
Nose Tip : 418 , 344 
Mouth : 421 , 370 
Left eye tragion : 377 , 319 
Right eye tragion : 510 , 327 

 The output image is saved in the same place as "images/test_image.jpg"
```

To visualize these facial keypoints on the image, the output image can be found at `images/test_image.jpg_drawn_out.jpg` path.

![1 face out of 6 detected](./assets/test_image.jpg_drawn_out.jpg)

### Similar results can be obtained by replacing `/path/to/image_file.jpg/` in both the commands with the other two images in the `./images` folder. 


#### This repository is created as a solution to pretest [LFX Mentorship 2023 01-Mar-May Challenge - for #2229 #2230](https://github.com/WasmEdge/WasmEdge/discussions/2230)