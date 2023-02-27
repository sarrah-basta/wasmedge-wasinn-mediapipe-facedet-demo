pub fn printf_face_detection_short(item : &(Vec<f32>, f32)){
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
}