#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
from tqdm import tqdm
import imagesize
import torch
import sys
import argparse
from PIL import Image
from scene.colmap_loader import qvec2rotmat
from utils.graphics_utils import focal2fov
import numpy as np
import json
from pathlib import Path
from dataclasses import dataclass
from multiprocessing.pool import ThreadPool

from mvs_modules.mvs_estimator import MvsEstimator

@dataclass
class CameraInfo:
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    depth: np.array
    bounds: np.array                            # 添加一组参数bounds,为图像的最大最小深度值，用于计算相机的视锥体
    image_path: str
    image_name: str
    width: int
    height: int
    timestamp: float = 0.0
    fl_x: float = -1.0
    fl_y: float = -1.0
    cx: float = -1.0
    cy: float = -1.0

    K: np.array = None
    mvs_depth: np.array = None
    mvs_mask: np.array = None
    fg_mask: np.array = None # for DTU evaluation
    mono_depth: np.array = None

def readCamerasFromTransforms2(path, transformsfile, white_background, extension=".png", time_duration=None, frame_ratio=1, dataloader=False):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        
    frames = contents["cam_info"]
    tbar = tqdm(range(len(frames)))
    def frame_read_fn(idx_frame):
        idx = idx_frame[0]
        frame = idx_frame[1]
        timestamp = frame.get('time', 0.0)
        bounds = np.array(frame['bounds'])              # 添加一组参数bounds,为图像的最大最小深度值，用于计算相机的视锥体
        if frame_ratio > 1:
            timestamp /= frame_ratio
        if time_duration is not None and 'time' in frame:
            if timestamp < time_duration[0] or timestamp > time_duration[1]:
                return

        cam_name = os.path.join(path, frame["file_path"] + extension)

        # # NeRF 'transform_matrix' is a camera-to-world transform
        c2w = np.array(frame["transform_matrix"])
        # # get the world-to-camera transform and set R, T
        w2c = np.linalg.inv(c2w)
        R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
        T = w2c[:3, 3]

        image_path = os.path.join(path, cam_name) 
        image_name = Path(cam_name).stem
        
        if not dataloader:
            with Image.open(image_path) as image_load:
                im_data = np.array(image_load.convert("RGBA"))

            bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            if norm_data[:, :, 3:4].min() < 1:
                arr = np.concatenate([arr, norm_data[:, :, 3:4]], axis=2)
                image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGBA")
            else:
                image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")

            width, height = image.size[0], image.size[1]
        else:
            image = np.empty(0)
            width, height = imagesize.get(image_path)
        
        if 'depth_path' in frame:
            depth_name = frame["depth_path"]
            if not extension in frame["depth_path"]:
                depth_name = frame["depth_path"] + extension
            depth_path = os.path.join(path, depth_name)
            depth = Image.open(depth_path).copy()
        else:
            depth = None
        tbar.update(1)
        if 'fl_x' in frame and 'fl_y' in frame and 'cx' in frame and 'cy' in frame:
            FovX = FovY = -1.0
            fl_x = frame['fl_x']
            fl_y = frame['fl_y']
            cx = frame['cx']
            cy = frame['cy']
            K = np.array([
                [fl_x, 0., cx],
                [0., fl_y, cy],
                [0., 0., 1.]
            ], dtype=np.float32)

            return CameraInfo(uid=idx, R=R, T=T, K=K, fg_mask=None, FovY=FovY, FovX=FovX, image=image, depth=depth, bounds=bounds, 
                        image_path=image_path, image_name=image_name, width=width, height=height, timestamp=timestamp,
                        fl_x=fl_x, fl_y=fl_y, cx=cx, cy=cy)
    
    with ThreadPool() as pool:      # 使用 map 方法将 frame_read_fn 函数应用到 frames 列表中的每个元素。
        cam_infos = pool.map(frame_read_fn, zip(list(range(len(frames))), frames)) 
        pool.close()
        pool.join()
        
    cam_infos = [cam_info for cam_info in cam_infos if cam_info is not None]
    
    return cam_infos


def readCamerasFromTransforms(path, transformsfile, white_background, extension=".png", time_duration=None, frame_ratio=1, dataloader=False):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        
    frames = contents["frames"]
    tbar = tqdm(range(len(frames)))
    def frame_read_fn(idx_frame):
        idx = idx_frame[0]
        frame = idx_frame[1]
        timestamp = frame.get('time', 0.0)
        bounds = np.array(frame['bounds'])              # 添加一组参数bounds,为图像的最大最小深度值，用于计算相机的视锥体
        if frame_ratio > 1:
            timestamp /= frame_ratio
        if time_duration is not None and 'time' in frame:
            if timestamp < time_duration[0] or timestamp > time_duration[1]:
                return

        cam_name = os.path.join(path, frame["file_path"] + extension)

        # # NeRF 'transform_matrix' is a camera-to-world transform
        c2w = np.array(frame["transform_matrix"])
        # # get the world-to-camera transform and set R, T
        w2c = np.linalg.inv(c2w)
        R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
        T = w2c[:3, 3]

        image_path = os.path.join(path, cam_name) # .replace('hdImgs_unditorted', 'hdImgs_unditorted_rgba').replace('.jpg', '.png')
        image_name = Path(cam_name).stem
        
        if not dataloader:
            with Image.open(image_path) as image_load:
                im_data = np.array(image_load.convert("RGBA"))

            bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            if norm_data[:, :, 3:4].min() < 1:
                arr = np.concatenate([arr, norm_data[:, :, 3:4]], axis=2)
                image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGBA")
            else:
                image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")

            width, height = image.size[0], image.size[1]
        else:
            image = np.empty(0)
            width, height = imagesize.get(image_path)
        
        if 'depth_path' in frame:
            depth_name = frame["depth_path"]
            if not extension in frame["depth_path"]:
                depth_name = frame["depth_path"] + extension
            depth_path = os.path.join(path, depth_name)
            depth = Image.open(depth_path).copy()
        else:
            depth = None
        tbar.update(1)
            
        if 'fl_x' in contents and 'fl_y' in contents and 'cx' in contents and 'cy' in contents:
            FovX = FovY = -1.0
            fl_x = contents['fl_x']
            fl_y = contents['fl_y']
            cx = contents['cx']
            cy = contents['cy']
            K = np.array([
                [fl_x, 0., cx],
                [0., fl_y, cy],
                [0., 0., 1.]
            ], dtype=np.float32)
            return CameraInfo(uid=idx, R=R, T=T, K=K, fg_mask=None, FovY=FovY, FovX=FovX, image=image, depth=depth, bounds=bounds, 
                        image_path=image_path, image_name=image_name, width=width, height=height, timestamp=timestamp,
                        fl_x=fl_x, fl_y=fl_y, cx=cx, cy=cy)
    
    with ThreadPool() as pool:      # 使用 map 方法将 frame_read_fn 函数应用到 frames 列表中的每个元素。
        cam_infos = pool.map(frame_read_fn, zip(list(range(len(frames))), frames)) 
        pool.close()
        pool.join()
        
    cam_infos = [cam_info for cam_info in cam_infos if cam_info is not None]
    
    return cam_infos

if __name__ == '__main__':
    parser = argparse.ArgumentParser() 
    parser.add_argument("--path", default="", help="input path to the camera parameters")
    parser.add_argument("--mvs_config", default="", help="")
    parser.add_argument("--dataset", default="", help="dynerf, techicolor")

    args = parser.parse_args()
    
    # path must end with / to make sure image path is relative
    if args.path[-1] != '/':
        args.path += '/'
    
    if args.dataset == "dynerf":
    
        print("Reading Training Transforms")    
        train_cam_infos = readCamerasFromTransforms(args.path, "transforms_train.json", white_background = False, extension = ".png", time_duration = [0, 10], frame_ratio=1.0, dataloader = False)
        
        save_path = os.path.join(args.path, "mvs")
        time_stamp =sorted(list(set([cam.timestamp for cam in train_cam_infos])))
        cam_infos = []

        for time in time_stamp:
            cam_infos = [cam for cam in train_cam_infos if cam.timestamp == time]
            mvs_idx = [4, 6, 13] # cut_beef train 05, 07, 16
            mvs_cam = [c for idx, c in enumerate(cam_infos) if idx in mvs_idx]
            
            print('Predicting MVS depth...')
            mvs_estimator = MvsEstimator(args.mvs_config)
            vertices, mvs_depths, masks = mvs_estimator.get_mvs_pts(mvs_cam, save_path)
            torch.cuda.empty_cache()

    elif args.dataset == "technicolor":
        
        print("Reading Training Transforms")    
        train_cam_infos = readCamerasFromTransforms2(args.path, "transforms_train.json", white_background = False, extension = ".png", time_duration = [0, 10], frame_ratio=1.0, dataloader = False)
        
        save_path = os.path.join(args.path, "mvs")
        time_stamp =sorted(list(set([cam.timestamp for cam in train_cam_infos])))
        cam_infos = []

        for time in time_stamp:
            cam_infos = [cam for cam in train_cam_infos if cam.timestamp == time]
            mvs_idx = [2, 8, 15] # Technicolor train cam 02, 08, 15
            mvs_cam = [c for idx, c in enumerate(cam_infos) if idx in mvs_idx]
            
            print('Predicting MVS depth...')
            mvs_estimator = MvsEstimator(args.mvs_config)
            vertices, mvs_depths, masks = mvs_estimator.get_mvs_pts(mvs_cam, save_path)
            torch.cuda.empty_cache()

