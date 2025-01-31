import pickle
import pickle5
import pandas as pd
import cv2
import torch
from tqdm import tqdm
import os
import yaml

from utils_preprocessing import *


traj_len = 20


def build_frame_list(data_raw):
    frame_dict = {}

    for i in range(0,len(data_raw),traj_len):
        frame_idx_list = data_raw[i:i+traj_len]['frame'].to_numpy().astype('float32')
        traj_xy = data_raw[i:i+traj_len][['x','y']].to_numpy().astype('float32')
        start_frame = int(frame_idx_list[0])
        sceneId = data_raw.iloc[i]['sceneId']
        # print(sceneId)
        if sceneId not in frame_dict.keys():
            frame_dict[sceneId] = []

        frame_dict[sceneId].append(start_frame)

    return frame_dict

def save_frames(frame_dict):
    for key in frame_dict.keys():

        scene, idx = key.split('_')
        video_name = 'videos/%s/video%s/video.mov'%(scene,idx)

        if not os.path.exists('keyframes/%s'%key):
            os.mkdir('keyframes/%s'%key)

        cap = cv2.VideoCapture(video_name)
        frame_idx = 0
        while True:
            # Capture frame-by-frame
            ret, frame = cap.read()
            # if frame is read correctly ret is True
            if not ret:
                break

            if frame_idx in frame_dict[key]:
                img_name = 'keyframes/%s/%07d.jpg'%(key,int(frame_idx))
                
                cv2.imwrite(img_name, frame)

                print(img_name) 
                # tmp = input()

            frame_idx = frame_idx + 1


if not os.path.exists('keyframes'):
    os.mkdir('keyframes')

with open('raw/test.pkl', 'rb') as f:
    data_raw = pickle5.load(f)

frame_dict = build_frame_list(data_raw)
save_frames(frame_dict)

with open('raw/train.pkl', 'rb') as f:
    data_raw = pickle5.load(f)

frame_dict = build_frame_list(data_raw)
save_frames(frame_dict)