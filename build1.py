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
print(config['video_path'])
print(config['input_shape'])
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
video = load_video(path = config['video_path'])

#checking array shape [n_frames, width, height, RGB]
print(video.shape)

sfa_layers = sksfa.HSFA(n_components=config['components'],
                        input_shape = config['input_shape'],
                        internal_batch_size=config['batch_size'],
                        noise_std = config['noise'],
                        final_degree= config['final_degree'],
                        layer_configurations=config['layerconfig'])
#Checking sfa layers
sfa_layers.summary()

#Training sfa layers (double check paper)
sfa_layers.fit(video)
#checking training file shape
print(video[0].shape)

#Extracting features with trained model
extracted_features = sfa_layers.transform(video)

#Creating plot
fig, ax = plt.subplots(2, sharex=True)
cutoff = 60
ax[0, 0].plot(output[:cutoff, 0])
ax[1, 0].plot(output[:cutoff, 1])

#Saving plot to local dir
plt.tight_layout()
fig.savefig('graph.png')