import sklearn as skl
import sksfa

import yaml
import random
import re
import os
import tempfile
import ssl
import cv2
import numpy as np
from matplotlib import pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, required=True, help='Path to config file')
args = parser.parse_args()

with open(args.config, 'r') as f:
    config = yaml.safe_load(f)

print(config)
def crop_center_square(frame):
  y, x = frame.shape[0:2]
  min_dim = min(y, x)
  start_x = (x // 2) - (min_dim // 2)
  start_y = (y // 2) - (min_dim // 2)
  return frame[start_y:start_y+min_dim,start_x:start_x+min_dim]

#Defining function to load video as array
def load_video(path, max_frames= config['fps']*config['length'], resize=(320,40)):
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
video = load_video(config['video_path'])

#checking array shape [n_frames, width, height, RGB]
print(video.shape)

sfa_layers = sksfa.HSFA(n_components=config['components'],
                        input_shape = config['input_shape'],
                        internal_batch_size=config['batch_size'],
                        noise_std = config['noise'],
                        final_degree= config['final_degree'],
                        layer_configurations=config['layerconfig'])


sfa_layers.fit(video)
print(video[0].shape)
extracted_features = sfa_layers.transform(video[4])
print(extracted_features)



