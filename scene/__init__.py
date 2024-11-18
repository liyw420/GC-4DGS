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
import torch
import random
import json
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON
from utils.data_utils import CameraDataset
from scene.cameras import PseudoCamera
from utils.pose_utils import generate_random_poses
from utils.helper_train import getfisheyemapper

class Scene:

    gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None, shuffle=True, resolution_scales=[1.0], num_pts=100_000, num_pts_ratio=1.0, time_duration=None, pcd_init="MVSFormer"):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians
        self.white_background = args.white_background

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}
        self.pseudo_cameras = {}

        if os.path.exists(os.path.join(args.source_path, "cameras_parameters.txt")):
            scene_info = sceneLoadTypeCallbacks["Technicolor"](args.source_path, args.white_background, args.eval, num_pts=num_pts, time_duration=time_duration, extension=args.extension, num_extra_pts=args.num_extra_pts, frame_ratio=args.frame_ratio, dataloader=args.dataloader, pcd_init=pcd_init)

        elif os.path.exists(os.path.join(args.source_path, "cam00.mp4")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval, num_pts=num_pts, time_duration=time_duration, extension=args.extension, num_extra_pts=args.num_extra_pts, frame_ratio=args.frame_ratio, dataloader=args.dataloader, pcd_init=pcd_init)
        
        elif os.path.exists(os.path.join(args.source_path, "cameras.txt")):
            scene_info = sceneLoadTypeCallbacks["Enerf_outdoor"](args.source_path, args.white_background, args.eval, num_pts=num_pts, time_duration=time_duration, extension=args.extension, num_extra_pts=args.num_extra_pts, frame_ratio=args.frame_ratio, dataloader=args.dataloader, pcd_init=pcd_init)

        else:
            assert False, "Could not recognize scene type!"

        if not self.loaded_iter:
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            # 合并全部摄像机信息并将其保存到一个名为 cameras.json 文件中, 对于dynerf，前300张图片为测试集，后5700张图片为训练集
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        # 把getNerfppNorm返回的结果的半径赋给cameras_extent，所有相机的中心点位置到最远camera的距离
        self.cameras_extent = scene_info.nerf_normalization["radius"]

        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args)


            total_timestamp = []                                                            # 查询全部图片的时间戳
            for i in self.train_cameras[resolution_scale]:
                total_timestamp.append(i.timestamp)
            total_timestamp = sorted(list(set(total_timestamp)))                            # 从列表中删除重复的元素
            
            pseudo_cams = []
            pseudo_poses = generate_random_poses(self.train_cameras[resolution_scale])      # 随机生成伪视角

            view = self.train_cameras[resolution_scale][0]
            for pose in pseudo_poses:
                for timestamp in total_timestamp:
                    pseudo_cams.append(PseudoCamera(
                        R=pose[:3, :3].T, T=pose[:3, 3], FoVx=view.FoVx, FoVy=view.FoVy,
                        width=view.image_width, height=view.image_height, timestamp=timestamp, cx=view.cx, cy=view.cy, fl_x=view.fl_x, fl_y=view.fl_y,
                    ))
            self.pseudo_cameras[resolution_scale] = pseudo_cams
            
        if args.loaded_pth:     # 运行render.py时从这里加载已经训练好的模型
            self.gaussians.restore(model_args=torch.load(args.loaded_pth)[0], training_args=None)
        else:
            if self.loaded_iter:
                # 加载初始点云
                self.gaussians.load_ply(os.path.join(self.model_path,
                                                            "point_cloud",
                                                            "iteration_" + str(self.loaded_iter),
                                                            "point_cloud.ply"))
            else:
                self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)

    def save(self, iteration):
        torch.save((self.gaussians.capture(), iteration), self.model_path + "/chkpnt" + str(iteration) + ".pth")

    def getTrainCameras(self, scale=1.0):
        return CameraDataset(self.train_cameras[scale].copy(), self.white_background)
        
    def getTestCameras(self, scale=1.0):
        return CameraDataset(self.test_cameras[scale].copy(), self.white_background)
    
    def getPseudoCameras(self, scale=1.0):
        if len(self.pseudo_cameras) == 0:
            return [None]
        else:
            return self.pseudo_cameras[scale]