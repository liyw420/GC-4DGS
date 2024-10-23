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
import cv2
import random
import torch
from torch import nn
import torchvision
import torch.nn.functional as F
from utils.loss_utils import l1_loss, l2_loss, ssim, msssim, loss_depth_smoothness, patch_norm_mse_loss, smooth_l1_loss
from utils.depth_utils import estimate_depth_MiDas, estimate_depth_DV2
from gaussian_renderer import render,render_for_depth
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state, knn, vis_depth
import uuid
from random import randint
from tqdm import tqdm
from utils.image_utils import psnr, easy_cmap
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from torchvision.utils import make_grid
import numpy as np
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig
from torch.utils.data import DataLoader
from torchmetrics.functional.regression import pearson_corrcoef

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint, debug_from,
             gaussian_dim, time_duration, num_pts, num_pts_ratio, rot_4d, force_sh_3d, batch_size):
    
    if dataset.frame_ratio > 1:
        time_duration = [time_duration[0] / dataset.frame_ratio,  time_duration[1] / dataset.frame_ratio]
    
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree, gaussian_dim=gaussian_dim, time_duration=time_duration, rot_4d=rot_4d, force_sh_3d=force_sh_3d, sh_degree_t=2 if pipe.eval_shfs_4d else 0, dist_thres=args.dist_thres)
    scene = Scene(dataset, gaussians, num_pts=num_pts, num_pts_ratio=num_pts_ratio, time_duration=time_duration)
    gaussians.training_setup(opt)
    
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0] # background is black 
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True) # create CUDA event for time recording.
    iter_end = torch.cuda.Event(enable_timing = True)
    
    best_psnr = 0.0
    ema_loss_for_log = 0.0
    ema_l1loss_for_log = 0.0
    ema_ssimloss_for_log = 0.0
    lambda_all = [key for key in opt.__dict__.keys() if key.startswith('lambda') and key!='lambda_dssim']
    for lambda_name in lambda_all:  # create new variables like "ema_+...+_for_log" with initialized value of 0.0
        vars()[f"ema_{lambda_name.replace('lambda_','')}_for_log"] = 0.0
    
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
        
    if pipe.env_map_res:
        env_map = nn.Parameter(torch.zeros((3,pipe.env_map_res, pipe.env_map_res),dtype=torch.float, device="cuda").requires_grad_(True))
        env_map_optimizer = torch.optim.Adam([env_map], lr=opt.feature_lr, eps=1e-15)
    else:
        env_map = None
        
    gaussians.env_map = env_map
        
    training_dataset = scene.getTrainCameras()  # 创建训练集对象和训练数据加载器
    training_dataloader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True, num_workers=12 if dataset.dataloader else 0, collate_fn=lambda x: x, drop_last=True)
     
    iteration = first_iter
    pseudo_stack = None
    selectviews = []
    while iteration < opt.iterations + 1:
        for batch_data in training_dataloader:
            iteration += 1
            if iteration > opt.iterations:
                break

            iter_start.record()
            gaussians.update_learning_rate(iteration)
            
            # Every 1000 its we increase the levels of SH up to a maximum degree
            if iteration % opt.sh_increase_interval == 0:
                gaussians.oneupSHdegree()
                
            # Render
            if (iteration - 1) == debug_from:
                pipe.debug = True
            
            batch_point_grad = []
            batch_visibility_filter = []
            batch_radii = []
            
            for batch_idx in range(batch_size):
                gt_image_and_depth, viewpoint_cam = batch_data[batch_idx]
                gt_image = gt_image_and_depth[0].cuda()
                viewpoint_cam = viewpoint_cam.cuda()

                render_pkg = render(viewpoint_cam, gaussians, pipe, background)     # 显存会发生变化
                image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
                
                # Image Loss
                Ll1 = l1_loss(image, gt_image)
                Lssim = 1.0 - ssim(image, gt_image)                                 # 显存会发生变化
                loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * Lssim

                # # depth loss: A. pearson_loss; B. l1_loss; C. global ordinal_loss; D. local patch loss; E. mvs depth loss; F. pseudo_view depth loss
                # rendered_depth = render_pkg["depth"][0]
                # norm_depth = (rendered_depth - rendered_depth.min()) / (rendered_depth.max() - rendered_depth.min())
                # dv2_depth = gt_image_and_depth[1].cuda()
                
                # # # A. pearson_loss
                # # depth = depth.reshape(-1, 1).squeeze(-1)
                # # gt_depth = gt_depth.reshape(-1, 1).squeeze(-1)
                # # depth_loss = (1 - pearson_corrcoef(gt_depth, depth))
                
                # # # B. l1_loss
                # # depth_loss = 0.1 * l1_loss(depth, gt_depth)

                # # C. global ordinal_loss
                # w, h = norm_depth.shape[1], norm_depth.shape[0]
                # base_tensor = torch.arange(0, w * h)
                # num_pairs = 2_200_000   # 1_300_000 dynerf dataset
                # if num_pairs > 2_200_000:
                #     random_indices_1 = torch.stack((torch.randperm(w * h)[:1_300_000], torch.randperm(w * h)[:1_300_000]), dim = 1)
                #     random_indices_2 = torch.stack((torch.randperm(w * h)[:(num_pairs - 1_300_000)], torch.randperm(w * h)[:(num_pairs - 1_300_000)]), dim = 1)
                #     random_pairs = torch.cat((base_tensor[random_indices_1], base_tensor[random_indices_2]), dim=0)
                
                # else:
                #     random_indices = torch.stack((torch.randperm(w * h)[:num_pairs], torch.randperm(w * h)[:num_pairs]), dim = 1)
                #     random_pairs = base_tensor[random_indices]
                
                # depth_pixel_1 = torch.stack((torch.div(random_pairs[:, 0], w, rounding_mode='trunc'), random_pairs[:, 0] % w), dim = 1)
                # depth_pixel_2 = torch.stack((torch.div(random_pairs[:, 0], w, rounding_mode='trunc'), random_pairs[:, 1] % w), dim = 1)
                
                # depth_1 = norm_depth[depth_pixel_1[:, 0], depth_pixel_1[:, 1]]
                # depth_2 = norm_depth[depth_pixel_2[:, 0], depth_pixel_2[:, 1]]
                # gt_depth_1 = dv2_depth[depth_pixel_1[:, 0], depth_pixel_1[:, 1]]
                # gt_depth_2 = dv2_depth[depth_pixel_2[:, 0], depth_pixel_2[:, 1]]
                
                # para = 1_000_000 
                # depth_loss_ordinal  = 0.05 * l1_loss(torch.tanh(para * (depth_1 - depth_2)), torch.sign(gt_depth_1 - gt_depth_2))

                # loss += depth_loss_ordinal

                # # D. local patch loss
                # depth_loss_patch = patch_norm_mse_loss(norm_depth[None, None, ...], dv2_depth[None, None, ...], 16, 0.0002)
                # loss += 0.1 * depth_loss_patch

                # # if iteration > 3000:
                # #     loss += 0.02 * loss_depth_smoothness(norm_depth[None, None, ...], dv2_depth[None, None, ...])
                
                # # E. mvs depth loss
                # mvs_depth = gt_image_and_depth[2].cuda()
                # mvs_mask = gt_image_and_depth[3].cuda()
                # depth_loss_mvs = smooth_l1_loss(rendered_depth * mvs_mask, mvs_depth * mvs_mask)
                # loss += 0.1 * depth_loss_mvs

                # # F. pseudo_view depth loss
                # if iteration % args.sample_pseudo_interval == 0 and iteration > args.start_sample_pseudo and iteration < args.end_sample_pseudo:
                #     if not pseudo_stack:
                #         pseudo_stack = scene.getPseudoCameras().copy()
                #     pseudo_cam = pseudo_stack.pop(randint(0, len(pseudo_stack) - 1))
                    
                #     render_pkg_pseudo = render(pseudo_cam, gaussians, pipe, background)
                #     rendered_depth_pseudo = render_pkg_pseudo["depth"][0]
                #     rendered_depth_pseudo = (rendered_depth_pseudo - rendered_depth_pseudo.min()) / (rendered_depth_pseudo.max() - rendered_depth_pseudo.min())
                #     # midas_depth_pseudo = estimate_depth_MiDas(render_pkg_pseudo["render"], mode='train')
                #     dv2_depth_pseudo = estimate_depth_DV2(render_pkg_pseudo["render"], input_size=600)
                    
                #     # torchvision.utils.save_image(rendered_depth_pseudo, "pseudo.png")
                #     # torchvision.utils.save_image(dv2_depth_pseudo, "dv2pseudo.png")

                #     # pearson_loss
                #     rendered_depth_pseudo = rendered_depth_pseudo.reshape(-1, 1).squeeze(-1)
                #     dv2_depth_pseudo = dv2_depth_pseudo.reshape(-1, 1).squeeze(-1)
                #     depth_loss_pseudo = (1 - pearson_corrcoef(rendered_depth_pseudo, -dv2_depth_pseudo)).mean()

                #     # # ordinal_loss
                #     # rendered_depth_pseudo_1 = rendered_depth_pseudo[depth_pixel_1[:, 0], depth_pixel_1[:, 1]]
                #     # rendered_depth_pseudo_2 = rendered_depth_pseudo[depth_pixel_2[:, 0], depth_pixel_2[:, 1]]
                #     # dv2_depth_pseudo_1 = dv2_depth_pseudo[depth_pixel_1[:, 0], depth_pixel_1[:, 1]]
                #     # dv2_depth_pseudo_2 = dv2_depth_pseudo[depth_pixel_2[:, 0], depth_pixel_2[:, 1]]
                
                #     # para = 1_000_000 
                #     # depth_loss_pseudo = l1_loss(torch.tanh(para * (rendered_depth_pseudo_1 - rendered_depth_pseudo_2)), torch.sign(dv2_depth_pseudo_1 - dv2_depth_pseudo_2))

                #     if torch.isnan(depth_loss_pseudo).sum() == 0:
                #         loss_scale = min((iteration - args.start_sample_pseudo) / 500., 1)
                #         loss += loss_scale * args.depth_pseudo_weight * depth_loss_pseudo

                # ###### opa mask Loss ######
                # if opt.lambda_opa_mask > 0:
                #     o = render_pkg['alpha'].clamp(1e-6, 1-1e-6)
                #     sky = 1 - o

                #     Lopa_mask = (- sky * torch.log(1 - o)).mean()

                #     lambda_opa_mask = opt.lambda_opa_mask
                #     loss = loss + lambda_opa_mask * Lopa_mask
                # ###### opa mask Loss ######
                
                # ###### rigid loss ######
                # if opt.lambda_rigid > 0:
                #     k = 20
                #     xyz_mean = gaussians.get_xyz
                #     xyz_cur =  xyz_mean #  + delta_mean
                #     idx, dist = knn(xyz_cur[None].contiguous().detach(), 
                #                     xyz_cur[None].contiguous().detach(), 
                #                     k)
                #     _, velocity = gaussians.get_current_covariance_and_mean_offset(1.0, gaussians.get_t + 0.1)
                #     weight = torch.exp(-100 * dist)
                #     vel_dist = torch.norm(velocity[idx] - velocity[None, :, None], p=2, dim=-1)
                #     Lrigid = (weight * vel_dist).sum() / k / xyz_cur.shape[0]
                #     loss = loss + opt.lambda_rigid * Lrigid
                # ########################
                
                # ###### motion loss ######
                # if opt.lambda_motion > 0:
                #     _, velocity = gaussians.get_current_covariance_and_mean_offset(1.0, gaussians.get_t + 0.1)
                #     Lmotion = velocity.norm(p=2, dim=1).mean()
                #     loss = loss + opt.lambda_motion * Lmotion
                # ########################

                loss = loss / batch_size
                loss.backward()                     # 显存会发生变化
                batch_point_grad.append(torch.norm(viewspace_point_tensor.grad[:,:2], dim=-1))
                batch_radii.append(radii)
                batch_visibility_filter.append(visibility_filter)

            if batch_size > 1:
                visibility_count = torch.stack(batch_visibility_filter,1).sum(1)
                visibility_filter = visibility_count > 0
                radii = torch.stack(batch_radii,1).max(1)[0]
                
                batch_viewspace_point_grad = torch.stack(batch_point_grad,1).sum(1)
                batch_viewspace_point_grad[visibility_filter] = batch_viewspace_point_grad[visibility_filter] * batch_size / visibility_count[visibility_filter]
                batch_viewspace_point_grad = batch_viewspace_point_grad.unsqueeze(1)
                
                if gaussians.gaussian_dim == 4:
                    batch_t_grad = gaussians._t.grad.clone()[:,0].detach()
                    batch_t_grad[visibility_filter] = batch_t_grad[visibility_filter] * batch_size / visibility_count[visibility_filter]
                    batch_t_grad = batch_t_grad.unsqueeze(1)
            else:
                if gaussians.gaussian_dim == 4:
                    batch_t_grad = gaussians._t.grad.clone().detach()
            
            iter_end.record()

            loss_dict = {"Ll1": Ll1,
                        "Lssim": Lssim}
            
            # loss_dict = {"Ll1": Ll1,
            #             "Lssim": Lssim,
            #             "Ldepth_ordinal": depth_loss_ordinal,
            #             "Ldepth_patch": depth_loss_patch,
            #             "Ldepth_mvs": depth_loss_mvs}

            with torch.no_grad():
                psnr_for_log = psnr(image, gt_image).mean().double()
                # Progress bar
                ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
                ema_l1loss_for_log = 0.4 * Ll1.item() + 0.6 * ema_l1loss_for_log
                ema_ssimloss_for_log = 0.4 * Lssim.item() + 0.6 * ema_ssimloss_for_log
                
                for lambda_name in lambda_all:
                    if opt.__dict__[lambda_name] > 0:
                        ema = vars()[f"ema_{lambda_name.replace('lambda_', '')}_for_log"]
                        vars()[f"ema_{lambda_name.replace('lambda_', '')}_for_log"] = 0.4 * vars()[f"L{lambda_name.replace('lambda_', '')}"].item() + 0.6*ema
                        loss_dict[lambda_name.replace("lambda_", "L")] = vars()[lambda_name.replace("lambda_", "L")]
                        
                if iteration % 10 == 0:
                    postfix = {"Loss": f"{ema_loss_for_log:.{7}f}",
                                            "PSNR": f"{psnr_for_log:.{2}f}",
                                            "Ll1": f"{ema_l1loss_for_log:.{4}f}",
                                            "Lssim": f"{ema_ssimloss_for_log:.{4}f}",}
                    
                    for lambda_name in lambda_all:
                        if opt.__dict__[lambda_name] > 0:
                            ema_loss = vars()[f"ema_{lambda_name.replace('lambda_', '')}_for_log"]
                            postfix[lambda_name.replace("lambda_", "L")] = f"{ema_loss:.{4}f}"
                            
                    progress_bar.set_postfix(postfix)
                    progress_bar.update(10)
                if iteration == opt.iterations:
                    progress_bar.close()

                # Log and save   test_psnr时显存会激增
                test_psnr = training_report(tb_writer, iteration, Ll1, Lssim, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background), loss_dict)
                if (iteration in testing_iterations):
                    if test_psnr >= best_psnr:
                        best_psnr = test_psnr
                        print("\n[ITER {}] Saving best checkpoint".format(iteration))
                        torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt_best.pth")
                        
                if (iteration in saving_iterations):
                    print("\n[ITER {}] Saving Gaussians".format(iteration))
                    scene.save(iteration)

                # Densification
                if iteration < opt.densify_until_iter and (opt.densify_until_num_points < 0 or gaussians.get_xyz.shape[0] < opt.densify_until_num_points):
                    # Keep track of max radii in image-space for pruning
                    gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                    if batch_size == 1:
                        gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter, batch_t_grad if gaussians.gaussian_dim == 4 else None)
                    else:
                        gaussians.add_densification_stats_grad(batch_viewspace_point_grad, visibility_filter, batch_t_grad if gaussians.gaussian_dim == 4 else None)
                         
                    if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                        size_threshold = 15 if iteration > opt.opacity_reset_interval else None
                        if iteration % 1000 == 0:
                            opt.densify_grad_threshold = opt.densify_grad_threshold * 0.82
                            # opt.densify_grad_t_threshold = opt.densify_grad_t_threshold * 0.82
                        gaussians.densify_and_prune(opt.densify_grad_threshold, opt.thresh_opa_prune, scene.cameras_extent, size_threshold, opt.densify_grad_t_threshold)
                    
                    if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                        gaussians.reset_opacity()
                
                # # depth guided densification
                # if (iteration >= 4000 and iteration % 2000 == 0):
                #     # selectviews = ['cam05','cam07','cam16']
                #     selectviews =[]

                #     if len(selectviews) > 0:

                #     # for batch_idx in range(batch_size):

                #         # 选取要进行 guided densification 的对应图片，每个视角只进行一次 guided densification
                #         gt_image_and_depth, viewpoint_cam = batch_data[0]
                #         gt_image = gt_image_and_depth[0].cuda()
                #         gt_depth = gt_image_and_depth[1].cuda()
                #         viewpoint_cam = viewpoint_cam.cuda()
                #         select_cam_name = viewpoint_cam.image_name[:5]
                #         print(select_cam_name)
                        
                #         if select_cam_name in selectviews:
                #             selectviews.remove(select_cam_name)
                #             render_pkg = render(viewpoint_cam, gaussians, pipe, background)
                #             image, depth = render_pkg["render"], render_pkg["depth"]
                            
                #             # 对渲染图片，真实图片和渲染深度进行patchify
                #             kernel_size, stride, dilation = 4, 4, 1

                #             image_patches = F.unfold(image[None, ...], kernel_size=kernel_size, stride=stride, dilation=dilation
                #                                     ).permute(0,2,1).view(-1, 3, kernel_size, stride)
                            
                #             gt_image_patches = F.unfold(gt_image[None, ...], kernel_size=kernel_size, stride=stride, dilation=dilation
                #                                         ).permute(0,2,1).view(-1, 3, kernel_size, stride)
                            
                #             # depth_patches = F.unfold(depth[None, ...], kernel_size=kernel_size, stride=stride, dilation=dilation
                #             #                          ).permute(0,2,1).view(-1, 1, kernel_size, stride)
                            
                #             # 计算patch之间的相似度,以此为依据选择不准确的patch进行 guided densification
                #             patch_similar = torch.tensor([ssim(image_patches, gt_image_patches, window_size=2) 
                #                                         for image_patches, gt_image_patches in zip(image_patches, gt_image_patches)])
                            
                #             sorted_patch_similar, _ = torch.sort(patch_similar, descending=True)
                #             threshold = sorted_patch_similar[int(0.9 * len(sorted_patch_similar))].item()
                #             error_mask = patch_similar < threshold

                #             for i in range(len(error_mask)):
                #                 if error_mask[i] == 1:
                #                     image_patches[i,:,:,:] = 0
                #                     gt_image_patches[i,:,torch.randint(int(kernel_size/2-1),int(kernel_size/2+1), (1,)),
                #                                     torch.randint(int(kernel_size/2-1),int(kernel_size/2+1), (1,))] = 2 # 获取patch选中像素的标记，选中像素的值为2，其余像素的取值范围为0-1
                #                     # depth_patches[i,:,:,:] = 0

                #             # 将selected patch重建为原始图像,并保存(检查selected patch是否正确)
                #             image_patches_selected = image_patches.view(image_patches.shape[0], -1).permute(1, 0).unsqueeze(0)
                #             output_image = F.fold(image_patches_selected, output_size= (gt_image.shape[1], gt_image.shape[2]), 
                #                                 kernel_size=kernel_size, stride=stride, dilation=dilation).squeeze(0)
                            
                #             gt_image_patches_selected = gt_image_patches.view(gt_image_patches.shape[0], -1).permute(1, 0).unsqueeze(0)
                #             output_gt_image = F.fold(gt_image_patches_selected, output_size= (gt_image.shape[1], gt_image.shape[2]), 
                #                                     kernel_size=kernel_size, stride=stride, dilation=dilation).squeeze(0)
                            
                #             # depth_patches_selected = depth_patches.view(depth_patches.shape[0], -1).permute(1, 0).unsqueeze(0)
                #             # output_depth = F.fold(depth_patches_selected, output_size= (depth.shape[1], depth.shape[2]), 
                #                                 #  kernel_size=kernel_size, stride=stride, dilation=dilation).squeeze(0)
                            
                #             torchvision.utils.save_image(image, os.path.join(scene.model_path,  "render_" + str(iteration) + str(select_cam_name) + ".png"))
                #             torchvision.utils.save_image(gt_image, os.path.join(scene.model_path,  "gt_" + str(iteration) + str(select_cam_name) + ".png"))
                #             torchvision.utils.save_image(output_image, os.path.join(scene.model_path,  "maskedrender_" + str(iteration) + str(select_cam_name) + ".png"))
                #             torchvision.utils.save_image(output_gt_image, os.path.join(scene.model_path,  "maskedgt_" + str(iteration) + str(select_cam_name) + ".png"))

                #             # depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255
                #             depth_map = vis_depth(depth[0].detach().cpu().numpy())
                #             # depth_map = depth[0].detach().cpu().numpy()
                #             cv2.imwrite(os.path.join(scene.model_path,  "renderdepth_" + str(iteration) + str(select_cam_name)+ ".png"), depth_map)

                #             # output_depth = (output_depth - output_depth.min()) / (output_depth.max() - output_depth.min()) * 255
                #             # output_depth_map = output_depth[0].detach().cpu().numpy()
                #             # cv2.imwrite(os.path.join(scene.model_path,  "maskeddepth_" + str(iteration) + ".png"), output_depth_map)

                #             # 每个被选中的patch进行guided densification得到一个点云。获得点云的像素坐标
                #             selected_pixel = torch.nonzero(output_gt_image[0] == 2)

                #             # 基于深度图进行 guided densification
                #             depth = depth.squeeze(0).detach()
                #             gt_depth = gt_depth.detach()
                #             new_points = gaussians.guided_densification(selected_pixel, viewpoint_cam, depth, gt_depth, gt_image,)
                        
                #         else: 
                #             break

                # Optimizer step
                if iteration < opt.iterations:
                    gaussians.optimizer.step()              # 显存会发生变化
                    gaussians.optimizer.zero_grad(set_to_none = True)
                    if pipe.env_map_res and iteration < pipe.env_optimize_until:
                        env_map_optimizer.step()
                        env_map_optimizer.zero_grad(set_to_none = True)

def prepare_output_and_logger(args):                        # 根据输入参数设置输出文件夹、记录配置参数，并创建一个 Tensorboard 写入器
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, Lssim, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, loss_dict=None):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/ssim_loss', Lssim.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)
        tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
        if loss_dict is not None:
            # if "Lrigid" in loss_dict:
            #     tb_writer.add_scalar('train_loss_patches/rigid_loss', loss_dict['Lrigid'].item(), iteration)
            # if "Ldepth" in loss_dict:
            #     tb_writer.add_scalar('train_loss_patches/depth_loss', loss_dict['Ldepth'].item(), iteration)
            # if "Ltv" in loss_dict:
            #     tb_writer.add_scalar('train_loss_patches/tv_loss', loss_dict['Ltv'].item(), iteration)
            # if "Lopa" in loss_dict:
            #     tb_writer.add_scalar('train_loss_patches/opa_loss', loss_dict['Lopa'].item(), iteration)
            # if "Lptsopa" in loss_dict:
            #     tb_writer.add_scalar('train_loss_patches/pts_opa_loss', loss_dict['Lptsopa'].item(), iteration)
            # if "Lsmooth" in loss_dict:
            #     tb_writer.add_scalar('train_loss_patches/smooth_loss', loss_dict['Lsmooth'].item(), iteration)
            # if "Llaplacian" in loss_dict:
            #     tb_writer.add_scalar('train_loss_patches/laplacian_loss', loss_dict['Llaplacian'].item(), iteration)
            if "Ldepth_ordinal" in loss_dict:
                tb_writer.add_scalar('train_loss_patches/depth_loss_ordinal', loss_dict['Ldepth_ordinal'].item(), iteration)
            if "Ldepth_patch" in loss_dict:
                tb_writer.add_scalar('train_loss_patches/depth_loss_patch', loss_dict['Ldepth_patch'].item(), iteration)
            if "Ldepth_mvs" in loss_dict:
                tb_writer.add_scalar('train_loss_patches/depth_loss_mvs', loss_dict['Ldepth_mvs'].item(), iteration)

    psnr_test_iter = 0.0
    # Report test and samples of training set
    if iteration in testing_iterations:
        validation_configs = ({'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]},
                              {'name': 'test', 'cameras' : [scene.getTestCameras()[idx] for idx in range(len(scene.getTestCameras()))]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                ssim_test = 0.0
                msssim_test = 0.0
                for idx, batch_data in enumerate(tqdm(config['cameras'])):
                    gt_image_and_depth, viewpoint = batch_data
                    gt_image = gt_image_and_depth[0].cuda()
                    viewpoint = viewpoint.cuda()
                    
                    render_pkg = renderFunc(viewpoint, scene.gaussians, *renderArgs)
                    image = torch.clamp(render_pkg["render"], 0.0, 1.0)
                    
                    depth = easy_cmap(render_pkg['depth'][0])
                    alpha = torch.clamp(render_pkg['alpha'], 0.0, 1.0).repeat(3,1,1)
                    if tb_writer and (idx < 5):
                        grid = [gt_image, image, alpha, depth]
                        grid = make_grid(grid, nrow=2)
                        tb_writer.add_images(config['name'] + "_view_{}/gt_vs_render".format(viewpoint.image_name), grid[None], global_step=iteration)
                            
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                    ssim_test += ssim(image, gt_image).mean().double()
                    msssim_test += msssim(image[None].cpu(), gt_image[None].cpu())
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras']) 
                ssim_test /= len(config['cameras'])     
                msssim_test /= len(config['cameras'])        
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - ssim', ssim_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - msssim', msssim_test, iteration)
                if config['name'] == 'test':
                    psnr_test_iter = psnr_test.item()
                    
    torch.cuda.empty_cache()
    return psnr_test_iter

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--config", type=str)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[5_000, 7_000, 10_000, 20_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--start_checkpoint", type=str, default = None)
    
    parser.add_argument("--gaussian_dim", type=int, default=3)
    parser.add_argument("--time_duration", nargs=2, type=float, default=[-0.5, 0.5])
    parser.add_argument('--num_pts', type=int, default=100_000)
    parser.add_argument('--num_pts_ratio', type=float, default=1.0)
    parser.add_argument("--rot_4d", action="store_true")
    parser.add_argument("--force_sh_3d", action="store_true")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--exhaust_test", action="store_true")
    
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
        
    cfg = OmegaConf.load(args.config)                       # 实现在配置文件（例如cut_roasted_beef.yaml）中定义一些参数，并将其合并至args对象中
    def recursive_merge(key, host): 
        if isinstance(host[key], DictConfig):
            for key1 in host[key].keys():
                recursive_merge(key1, host[key])
        else:
            assert hasattr(args, key), key
            setattr(args, key, host[key])
    for k in cfg.keys():
        recursive_merge(k, cfg)
        
    if args.exhaust_test:                                   # 每迭代500次，即进行测试集上的评估
        args.test_iterations = args.test_iterations + [i for i in range(0,op.iterations,500)]
    
    setup_seed(args.seed)                                   # 设置随机数生成器的种子
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)                                  

    torch.autograd.set_detect_anomaly(args.detect_anomaly)  # 不启用计算图的异常监测
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.start_checkpoint, args.debug_from,
             args.gaussian_dim, args.time_duration, args.num_pts, args.num_pts_ratio, args.rot_4d, args.force_sh_3d, args.batch_size)

    # All done
    print("\nTraining complete.")
