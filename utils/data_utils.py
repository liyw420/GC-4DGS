import os
import torch
from torchvision.utils import save_image
from torch.utils.data import Dataset
from torchvision import datasets
from utils.general_utils import PILtoTorch
from PIL import Image
import numpy as np

class CameraDataset(Dataset):
    
    def __init__(self, viewpoint_stack, white_background):
        self.viewpoint_stack = viewpoint_stack
        self.bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])
        
    def __getitem__(self, index):
        viewpoint_cam = self.viewpoint_stack[index]
        if viewpoint_cam.meta_only:
            with Image.open(viewpoint_cam.image_path) as image_load:                                    # 这里是读取图像的代码
                im_data = np.array(image_load.convert("RGBA"))
            norm_data = im_data / 255.0
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + self.bg * (1 - norm_data[:, :, 3:4])
            image_load = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")

            resized_image_rgb = PILtoTorch(image_load, viewpoint_cam.resolution)
            
            viewpoint_image = resized_image_rgb[:3, ...].clamp(0.0, 1.0)
            if resized_image_rgb.shape[1] == 4:
                gt_alpha_mask = resized_image_rgb[3:4, ...]
                viewpoint_image *= gt_alpha_mask
            else:
                viewpoint_image *= torch.ones((1, viewpoint_cam.image_height, viewpoint_cam.image_width))
            
            path_parts = viewpoint_cam.image_path.split(os.sep)
            path_parts[-3] = '3_views'
            new_image_path = os.sep.join(path_parts)
            dv2_depth_path = new_image_path.replace("images", "depths").replace('.png', '_mde_depth.npy')
            # dv2_depth_path = viewpoint_cam.image_path.replace("images", "depths").replace('.png', '_mde_depth.npy') # dynerf/technicolor dataset
            # dv2_depth_path = viewpoint_cam.image_path.replace("images_half", "depths").replace('.png', '_mde_depth.npy') # enerf dataset
            if os.path.exists(dv2_depth_path):
                dv2_depth_array = np.load(dv2_depth_path)
                viewpoint_dv2_depth = torch.from_numpy(dv2_depth_array)
            else:
                viewpoint_dv2_depth = torch.zeros((viewpoint_cam.image_height, viewpoint_cam.image_width))
            
            mvs_depth_path = viewpoint_cam.image_path.replace("images", "mvs").replace('.png', '_depth.npy')
            # mvs_depth_path = viewpoint_cam.image_path.replace("images_half", "mvs").replace('.png', '_depth.npy')
            if os.path.exists(mvs_depth_path):
                mvs_depth_array = np.load(mvs_depth_path)
                viewpoint_mvs_depth = torch.from_numpy(mvs_depth_array)
            else:
                viewpoint_mvs_depth = torch.zeros((viewpoint_cam.image_height, viewpoint_cam.image_width))

            mask_path = viewpoint_cam.image_path.replace("images", "mvs").replace('.png', '_mask.npy')
            # mask_path = viewpoint_cam.image_path.replace("images_half", "mvs").replace('.png', '_mask.npy')
            if os.path.exists(mask_path):
                mvs_mask_array = np.load(mask_path)
                viewpoint_mvs_mask = torch.from_numpy(mvs_mask_array)
            else:
                viewpoint_mvs_mask = torch.zeros((viewpoint_cam.image_height, viewpoint_cam.image_width))

        else:
            viewpoint_image = viewpoint_cam.image
            
        return [viewpoint_image, viewpoint_dv2_depth, viewpoint_mvs_depth, viewpoint_mvs_mask], viewpoint_cam
    
    def __len__(self):
        return len(self.viewpoint_stack)
    
