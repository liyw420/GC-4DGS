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
import copy
import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, OptimizationParams, get_combined_args
from utils.pose_utils import generate_spiral_path_4dgs
from utils.graphics_utils import getWorld2View2
from utils.general_utils import vis_depth
from gaussian_renderer import GaussianModel
import numpy as np
import cv2
import open3d as o3d
from scripts.pre_dynerf.n3v2blender import closest_point_2_lines, rotmat
import time

def render_set(model_path, train_or_test, loaded_pth, views, gaussians, pipeline, background):
    
    if train_or_test in ["train", "test"]:
        model_name = loaded_pth.split('/')[-1].split('.')[0]
        render_path = os.path.join(model_path, train_or_test, model_name, "renders")
        depth_path = os.path.join(model_path, train_or_test, model_name, "depth")
        gts_path = os.path.join(model_path, train_or_test, model_name, "gt")

        makedirs(render_path, exist_ok=True)
        makedirs(gts_path, exist_ok=True)
        makedirs(depth_path, exist_ok=True)
        t_all = []

        for idx, view in enumerate(tqdm(views, desc="Rendering progress")):

            start = time.time()
            output = render(view[1].cuda(), gaussians, pipeline, background)                   # 渲染输出
            end = time.time()
            t_all.append(end - start)

            rendered_image = output["render"]
            gt = view[0][0][0:3, :, :]                                                         # Ground Truth图片
            torchvision.utils.save_image(rendered_image, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
            torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
            
            depth = output["depth"]
            depth_map = vis_depth(depth[0].detach().cpu().numpy())
            cv2.imwrite(os.path.join(depth_path,'{0:05d}'.format(idx) + ".png"), depth_map)

            # 保存点云
            if idx == 0:
                mean3d = output["mean3d"].detach().cpu().numpy()
                # colors = output["colors"].detach().cpu().numpy()
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(mean3d)
                # pcd.colors = o3d.utility.Vector3dVector(colors)
                o3d.io.write_point_cloud(os.path.join(model_path, train_or_test, model_name,"pcd.ply"), pcd)
            
        FPS = 1 / (sum(t_all) / len(t_all))
        print(f"FPS: {FPS}")

    else:
        model_name = loaded_pth.split('/')[-1].split('.')[0]
        render_path = os.path.join(model_path, train_or_test, model_name, "renders")
        depth_path = os.path.join(model_path, train_or_test, model_name, "depth")

        makedirs(render_path, exist_ok=True)
        makedirs(depth_path, exist_ok=True)

        for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
            output = render(view.cuda(), gaussians, pipeline, background)                # 渲染输出
            rendered_image = output["render"]
            torchvision.utils.save_image(rendered_image, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
            
            depth = output["depth"]
            depth_map = vis_depth(depth[0].detach().cpu().numpy())
            cv2.imwrite(os.path.join(depth_path,'{0:05d}'.format(idx) + ".png"), depth_map)

def render_video(source_path, model_path, loaded_pth, views, gaussians, pipeline, background, fps=30):
    
    model_name = loaded_pth.split('/')[-1].split('.')[0]
    render_path = os.path.join(model_path, 'video', model_name, 'renders')
    depth_path = os.path.join(model_path, 'video', model_name, 'depth')
    makedirs(render_path, exist_ok=True)
    makedirs(depth_path, exist_ok=True)

    view = copy.deepcopy(views[0])

    if source_path.find('N3V') != -1:
        ## 以下对poses_bounds.npy中的数据处理copy自scripts/n3v2blender.py
        poses_bounds = np.load(source_path + '/poses_bounds.npy')
        N = poses_bounds.shape[0]
        poses = poses_bounds[:, :15].reshape(-1, 3, 5) # (N, 3, 5)
        bounds = poses_bounds[:, -2:] # (N, 2)
        poses = np.concatenate([poses[..., 1:2], poses[..., 0:1], -poses[..., 2:3], poses[..., 3:4]], -1)
        last_row = np.tile(np.array([0, 0, 0, 1]), (len(poses), 1, 1)) # (N, 1, 4)
        poses = np.concatenate([poses, last_row], axis=1) # (N, 4, 4) 
        poses[:, 0:3, 1] *= -1
        poses[:, 0:3, 2] *= -1
        poses = poses[:, [1, 0, 2, 3], :] # swap y and z
        poses[:, 2, :] *= -1 # flip whole world upside down
        up = poses[:, 0:3, 1].sum(0)
        up = up / np.linalg.norm(up)
        R = rotmat(up, [0, 0, 1]) # rotate up vector to [0,0,1]
        R = np.pad(R, [0, 1])
        R[-1, -1] = 1
        poses = R @ poses
        totw = 0.0
        totp = np.array([0.0, 0.0, 0.0])
        for i in range(N):
            mf = poses[i, :3, :]
            for j in range(i + 1, N):
                mg = poses[j, :3, :]
                p, w = closest_point_2_lines(mf[:,3], mf[:,2], mg[:,3], mg[:,2])
                #print(i, j, p, w)
                if w > 0.01:
                    totp += p * w
                    totw += w
        totp /= totw
        print(f'[INFO] totp = {totp}')
        poses[:, :3, 3] -= totp
        avglen = np.linalg.norm(poses[:, :3, 3], axis=-1).mean()
        poses[:, :3, 3] *= 4.0 / avglen
        ## 以上对poses_bounds.npy中的数据处理copy自scripts/n3v2blender.py

        render_poses = generate_spiral_path_4dgs(poses, bounds)
        
    gt_image_and_depth = view[0]
    rendered_view = view[1].cuda()
    size = (gt_image_and_depth[0].shape[2], gt_image_and_depth[0].shape[1])
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    final_video = cv2.VideoWriter(os.path.join(render_path, 'final_video.mp4'), fourcc, fps, size)

    for idx, pose in enumerate(tqdm(render_poses, desc="Rendering progress")):
        rendered_view.world_view_transform = torch.tensor(getWorld2View2(pose[:3, :3].T, pose[:3, 3], np.array([0.0, 0.0, 0.0]), 1.0)).transpose(0, 1).cuda()
        rendered_view.full_proj_transform = (rendered_view.world_view_transform.unsqueeze(0).bmm(rendered_view.projection_matrix.unsqueeze(0))).squeeze(0)
        rendered_view.camera_center = rendered_view.world_view_transform.inverse()[3, :3]
        rendered_view.timestamp = idx / fps
        rendering = render(rendered_view, gaussians, pipeline, background)

        img = torch.clamp(rendering["render"], min=0., max=1.)
        torchvision.utils.save_image(img, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        video_img = (img.permute(1, 2, 0).detach().cpu().numpy() * 255.).astype(np.uint8)[..., ::-1]
        final_video.write(video_img)

        depth = rendering ["depth"]
        depth_map = vis_depth(depth[0].detach().cpu().numpy()) 
        cv2.imwrite(os.path.join(depth_path,'{0:05d}'.format(idx) + ".png"), depth_map)

    final_video.release()

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool,
                skip_pseudo:bool, skip_video: bool, mode: str, pcd_init: str):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree, gaussian_dim=4, time_duration=[0.0, 10.0], rot_4d=True, force_sh_3d=False, sh_degree_t=2 if pipeline.eval_shfs_4d else 0)
        scene = Scene(dataset, gaussians, num_pts=300000, num_pts_ratio=1.0, time_duration=[0.0, 10.0], shuffle=False, pcd_init = pcd_init)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]   # background is black
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if mode == "render":
            render_func = render_set
        # elif mode == "time":
        #     render_func = interpolate_time
        # elif mode == "view":
        #     render_func = interpolate_view
        # elif mode == "video":
        #     render_func = interpolate_poses
        # elif mode == "original":
        #     render_func = interpolate_view_original
        # else:
        #     render_func = interpolate_all

        # if not skip_train:
        #      render_func(dataset.model_path, "train", dataset.loaded_pth, scene.getTrainCameras(), gaussians, pipeline, background)

        if not skip_test:
            render_func(dataset.model_path, "test", dataset.loaded_pth, scene.getTestCameras(), gaussians, pipeline, background)
        
        # if not skip_pseudo:
        #     render_func(dataset.model_path, "pseudo", dataset.loaded_pth, scene.getPseudoCameras()[::(300+1)], gaussians, pipeline, background, pcd_init)
        
        # if not skip_video:
        #     render_video(dataset.source_path, dataset.model_path, dataset.loaded_pth, scene.getTestCameras(), gaussians, pipeline, background, pcd_init)            

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    op = OptimizationParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--skip_video", action="store_true")
    parser.add_argument("--skip_pseudo", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--mode", default='render', choices=['render', 'time', 'view', 'all', 'pose', 'original'])
    parser.add_argument("--pcd_init", type=str, default="COLMAP")
    args = get_combined_args(parser)
    
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, args.skip_pseudo, args.skip_video, args.mode, args.pcd_init)