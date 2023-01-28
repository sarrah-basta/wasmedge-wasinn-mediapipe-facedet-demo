import time
import sys
import cv2
import tkinter as tk
import numpy as np
from math import *

# to run command `python drawDetections.py "output_facial_key.bin" "image.jpg" `

# add input image path here
imagePath = sys.argv[2]
img = cv2.imread(imagePath, cv2.IMREAD_COLOR)

data = np.fromfile(sys.argv[1], dtype=np.uint32)
data_bytes= data.tobytes()
key = []
# print("integerrrr :", int.from_bytes(resized_data[4:8], byteorder='big'))
for i in range(0,64,4):
    key.append( int.from_bytes(data_bytes[i:i+4], byteorder='big'))
# bounding box co-ordinates are added here
cv2.rectangle(img, (key[0], key[1]), (key[2], key[3]), (22, 22, 250), 4)
# add facial keypoint co-ordinates here, in order : 
#            left eye,   right eye,  nose tip,   mouth,      left eye tragion, right eye tragion 
# keypoints = [[398, 311], [460, 314], [417, 344], [421, 370], [378, 318],       [513, 326]]
for i in range(4,15,2):
    xKeypoint = key[i]
    yKeypoint = key[i+1]
    cv2.circle(img,(xKeypoint,yKeypoint), 6, (251, 0, 249), -1)

area = 0.40

h, w = img.shape[:2]
root = tk.Tk()
screen_h = root.winfo_screenheight()
screen_w = root.winfo_screenwidth()
vector = sqrt(area)
window_h = screen_h * vector
window_w = screen_w * vector

if h > window_h or w > window_w:
    if h / window_h >= w / window_w:
        multiplier = window_h / h
    else:
        multiplier = window_w / w
    img = cv2.resize(img, (0, 0), fx=multiplier, fy=multiplier)

cv2.imshow("Resized_Window", img)
# cv2.imshow("detection results plotted using python", img)
cv2.waitKey(0)