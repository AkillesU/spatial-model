import sklearn as skl
import sksfa

import random
import re
import os
import tempfile
import ssl
import cv2
import numpy as np
from matplotlib import pyplot as plt

fps = 30
length = 5
video_path = "C:/Users/arech/Documents/PSYCHOLOGY/Love Lab/spatial-project/video1.mp4"

#Defining the parameters for a frame
def crop_center_square(frame):
  y, x = frame.shape[0:2]
  min_dim = min(y, x)
  start_x = (x // 2) - (min_dim // 2)
  start_y = (y // 2) - (min_dim // 2)
  return frame[start_y:start_y+min_dim,start_x:start_x+min_dim]

#Defining function to load video as array
def load_video(path, max_frames=fps*length, resize=(320,40)):
  cap = cv2.VideoCapture(path)
  frames = []
  try:
    while True:
      ret, frame = cap.read()
      if not ret:
        break
      frame = crop_center_square(frame)
      frame = cv2.resize(frame, resize)
      frame = frame[:, :, [2, 1, 0]]
      frames.append(frame)

      if len(frames) == max_frames:
        break
  finally:
    cap.release()
  return np.array(frames) / 255.0

#loading video as array from "video_path"
video = load_video(video_path)

#checking array shape [n_frames, width, height, RGB]
print(video.shape)
#plotting first frame
#plt.imshow(video[0], interpolation="nearest")
#plt.show()

input_shape = [320,40,3]
noise = 0.05
final_degree = 2
layerconfig = [(10,10,5,5,441,2), (7,3,4,4,30,2), (15,2,1,1,1,2)]
layers = 3
batch_size = 32

sfa_layers = sksfa.HSFA(n_components=layers,
                        input_shape = input_shape,
                        internal_batch_size=batch_size,
                        noise_std = noise,
                        final_degree= final_degree,
                        layer_configurations=layerconfig)


sfa_layers.fit(video)
print(video[0].shape)
extracted_features = sfa_layers.transform(video[4])
print(extracted_features)