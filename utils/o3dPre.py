import os
import argparse
import glob
import open3d as o3d

import numpy as np
import json
import sys
import math
import shutil
import sqlite3


# 点云读取
parser = argparse.ArgumentParser() 
parser.add_argument("--mvs_input", default="", help="")
parser.add_argument("--output_path", default="", help="")
parser.add_argument("--colmap", default="False", help="")
parser.add_argument("--mvsformer", default="True", help="")
args = parser.parse_args()

if args.colmap == "True":
    pcd = o3d.io.read_point_cloud(os.path.join(args.mvs_input, 'points3d_colmap.ply'))

    # # 点云过滤
    # print("Statistical oulier removal")
    # filtered_ptc,_ = pcd.remove_statistical_outlier(nb_neighbors=3,std_ratio=0.01)

    # 点云随机降采样
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)
    num_points = len(points)
    if num_points >= 250000:
        sample_ratio = 250000 / num_points
    else:
        sample_ratio = 1.0

    num_sampled_points = int(num_points * sample_ratio)
    indices = np.random.choice(num_points, num_sampled_points, replace=False)

    sampled_points = points[indices]
    sampled_colors = colors[indices]

    downsampled_pcd = o3d.geometry.PointCloud()
    downsampled_pcd.points = o3d.utility.Vector3dVector(sampled_points)
    downsampled_pcd.colors = o3d.utility.Vector3dVector(sampled_colors)

    # # coffee_martini_MVS
    # points = np.asarray(downsampled_pcd.points)
    # colors = np.asarray(downsampled_pcd.colors)
    # condition = np.logical_not(np.logical_and.reduce((
    #     points[:, 0] > 11.8591,
    #     points[:, 1] < -9.9617,
    #     points[:, 2] < 15.3459
    # )))

    # filtered_points = points[condition]
    # filtered_colors = colors[condition]

    # filtered_point_cloud = o3d.geometry.PointCloud()
    # filtered_point_cloud.points = o3d.utility.Vector3dVector(filtered_points)
    # filtered_point_cloud.colors = o3d.utility.Vector3dVector(filtered_colors)

    # 保存合并后的点云
    # output_path = os.path.join('/media/vincent/HDD-02/fs4dgs/data/N3V/cut_roasted_beef_MVS', 'points3d.ply')
    output_path = os.path.join(args.output_path, 'points3d_colmap.ply')
    o3d.io.write_point_cloud(output_path, downsampled_pcd)







if args.mvsformer == "True":
    pcd = o3d.io.read_point_cloud(os.path.join(args.mvs_input, 'mvs/mvs_input_0.0.ply'))

    # # 点云过滤
    # print("Statistical oulier removal")
    # filtered_ptc,_ = pcd.remove_statistical_outlier(nb_neighbors=3,std_ratio=0.01)

    # 点云随机降采样
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)
    num_points = len(points)
    if num_points >= 250000:
        sample_ratio = 250000 / num_points
    else:
        sample_ratio = 1.0

    num_sampled_points = int(num_points * sample_ratio)
    indices = np.random.choice(num_points, num_sampled_points, replace=False)

    sampled_points = points[indices]
    sampled_colors = colors[indices]

    downsampled_pcd = o3d.geometry.PointCloud()
    downsampled_pcd.points = o3d.utility.Vector3dVector(sampled_points)
    downsampled_pcd.colors = o3d.utility.Vector3dVector(sampled_colors)

    # coffee_martini, flame_salmon
    scene_name = os.path.basename(os.path.dirname(args.mvs_input))
    if scene_name == 'coffee_martini' or 'flame_salmon':
    # if scene_name == 'coffee_martini':
        points = np.asarray(downsampled_pcd.points)
        colors = np.asarray(downsampled_pcd.colors)
        condition = np.logical_not(np.logical_and.reduce((
            points[:, 0] > 11.8591,
            points[:, 1] < -9.9617,
            points[:, 2] < 15.3459
        )))

        filtered_points = points[condition]
        filtered_colors = colors[condition]

        downsampled_pcd = o3d.geometry.PointCloud()
        downsampled_pcd.points = o3d.utility.Vector3dVector(filtered_points)
        downsampled_pcd.colors = o3d.utility.Vector3dVector(filtered_colors)

    # 保存合并后的点云
    output_path = os.path.join(args.output_path, 'points3d_mvs.ply')
    o3d.io.write_point_cloud(output_path, downsampled_pcd)