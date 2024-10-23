import torch
import os
import numpy as np
from PIL import Image
from depth_utils import estimate_depth
from general_utils import vis_depth, PILtoTorch
import cv2



def get_png_files(folder_path):
    files = os.listdir(folder_path)
    png_files = sorted([file for file in files if file.endswith('.png')])
    return png_files

folder_path = "/home/vincent/Research/code/fudan4dgs/data/N3V/cut_roasted_beef_sparse_view"

image_names = get_png_files(os.path.join(folder_path, "images"))
image_paths = []
for image_name in image_names:
    image_paths.append(os.path.join(folder_path, "images", image_name))

for image_path in image_paths:
    with Image.open(image_path) as image_load:
        im_data = np.array(image_load.convert("RGBA"))

    bg = np.array([0, 0, 0])

    norm_data = im_data / 255.0
    arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
    if norm_data[:, :, 3:4].min() < 1:
        arr = np.concatenate([arr, norm_data[:, :, 3:4]], axis=2)
        image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGBA")
    else:
        image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")

    resized_image_rgb = PILtoTorch(image, (1352, 1014))
    gt_image = resized_image_rgb[:3, ...]
    
    # depth = estimate_depth(gt_image.cuda()).cpu().numpy()


    depth_map = vis_depth(depth)
    cv2.imwrite(os.path.join(folder_path,"depths", os.path.basename(image_path)), depth_map)