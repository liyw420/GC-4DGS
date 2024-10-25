# FS4DGS: Few-shot Dynamic Novel View Synthesis using Gaussian Splatting

**This repository is the official implementation of "FS4DGS: Few-shot Dynamic Novel View Synthesis using Gaussian Splatting".** In this paper, we propose XXX.

## Get started

### Environment

The hardware and software requirements are the same as those of the [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting), which this code is built upon. To setup the environment, please run the following command:

```shell
git clone https://github.com/liyw420/FS4DGS
cd FS4DGS
conda env create --file environment4dgs.yml
conda activate fs4dgs
```

### Data preparation

**DyNeRF dataset:**

Download the [Neural 3D Video dataset](https://github.com/facebookresearch/Neural_3D_Video) and extract each scene to `data/N3V`. After that, preprocess the raw video by executing:

```shell
python scripts/n3v2blender.py data/N3V/$scene_name
```

**Technicolor dataset:**

Please reach out to the authors of the paper "Dataset and Pipeline for Multi-View Light-Field Video" for access to the Technicolor dataset. </br> Our codebase expects the following directory structure for this dataset before preprocessing:
```
<location>
|---Fabien
|   |---Fabien_undist_<00257>_<08>.png
|   |---Fabien_undist_<.....>_<..>.png
|---Birthday
```
Then run the following command to preprocess the dataset. </br>
```
python scripts/pre_technicolor/pre_technicolor.py --videopath <location>/<scene>
```
Convert COLMAP version of camera parameters to NeRF version of camera parameters (obtain poses_bounds.npy). </br> Before this, set PYTHONPATH, otherwise errors will occur, hope to fix it in the future
```
export PYTHONPATH=<.....>/FS4DGS 
python scripts/pre_technicolor/imgs2poses.py --match_type exhaustive_matcher --scenedir <location>/<scene>
```
Then obtain the camera information of both training set and testing test (transforms_test.json & transforms_train.json):
```
python scripts/pre_technicolor/pose2MVS.py --path <location>/<scene>
```
Download the [pre-trained model](https://onedrive.live.com/?redeem=aHR0cHM6Ly8xZHJ2Lm1zL3UvcyFBaDJWa1VMbWtpcVByeUhfVGwyUFVTNklzODMxP2U9QmdDdU9Z&id=8F2A92E64291951D%216049&cid=8F2A92E64291951D) of MVSFormer (only MVSFormer.zip and MVSFormer-Blended.zip are enough). Then unzip them and put them in this directory: 
```
<.....>/FS4DGS/mvs_modules/pretrained/<.....>
```
Then use MVSFormer to obtain the point clouds, depthmaps, and maskmaps.
```
python mvs2points.py --path <location>/<scene> --mvs_config <.....>/fs4dgs/mvs_modules/configs/config_mvsformer.json --dataset technicolor
```
Point cloud downsampling and outlier filtering.
```
python scripts/pre_technicolor/o3dPre.py ----mvs_input <location>/<scene>/mvs --downsampling_rate 0.1 --output_path <location>/<scene>
```
Training on technicolor dataset:
```
python train.py --config configs/lightfield/birthday_MVS.yaml
```
Rendering on technicolor dataset:
```
python render.py --model_path output/lightfield/Birthday/ --loaded_pth output/lightfield/Birthday/chkpnt_best.pth
```
Evaluation on technicolor dataset:
```
python metrics.py --model_path output/lightfield/Birthday/
```

**Using DepthAnythingV2 for Monocular Depth Estimation:**</br>
Git clone the repository of [DepthAnythingV2 (DV2)](https://github.com/DepthAnything/Depth-Anything-V2/tree/31dc97708961675ce6b3a8d8ffa729170a4aa273), then put them in this directory:
```
<.....>/FS4DGS/utils/DV2/<.....>
```
Use the metrics-depth version of DV2, download the checkpoints [here](https://github.com/DepthAnything/Depth-Anything-V2/tree/31dc97708961675ce6b3a8d8ffa729170a4aa273/metric_depth), choosing **Depth-Anything-V2-Large**, training on **Indoor(Hypersim)**. The following code, named **DV2run.py** should replace the original **<.....>/DV2/metric_depth/run.py**:
```
import argparse
import cv2
import glob
import matplotlib
import numpy as np
import os
import torch
from util.general_utils import vis_depth, PILtoTorch

from depth_anything_v2.dpt import DepthAnythingV2


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Depth Anything V2 Metric Depth Estimation')
    
    parser.add_argument('--img-path', type=str)
    parser.add_argument('--input-size', type=int, default=518)
    parser.add_argument('--outdir', type=str, default='./vis_depth')
    
    parser.add_argument('--encoder', type=str, default='vitl', choices=['vits', 'vitb', 'vitl', 'vitg'])
    parser.add_argument('--load-from', type=str, default='checkpoints/depth_anything_v2_metric_hypersim_vitl.pth')
    parser.add_argument('--max-depth', type=float, default=20)
    
    parser.add_argument('--save-numpy', dest='save_numpy', action='store_true', help='save the model raw output')
    parser.add_argument('--pred-only', dest='pred_only', action='store_true', help='only display the prediction')
    parser.add_argument('--grayscale', dest='grayscale', action='store_true', help='do not apply colorful palette')
    
    args = parser.parse_args()
    
    DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }
    
    depth_anything = DepthAnythingV2(**{**model_configs[args.encoder], 'max_depth': args.max_depth})
    depth_anything.load_state_dict(torch.load(args.load_from, map_location='cpu'))
    depth_anything = depth_anything.to(DEVICE).eval()
    
    if os.path.isfile(args.img_path):
        if args.img_path.endswith('txt'):
            with open(args.img_path, 'r') as f:
                filenames = f.read().splitlines()
        else:
            filenames = [args.img_path]
    else:
        filenames = glob.glob(os.path.join(args.img_path, '**/*'), recursive=True)
    
    os.makedirs(args.outdir, exist_ok=True)
    
    cmap = matplotlib.colormaps.get_cmap('Spectral')
    
    filenames.sort()
    for k, filename in enumerate(filenames):
        print(f'Progress {k+1}/{len(filenames)}: {filename}')
        
        raw_image = cv2.imread(filename)
        
        depth = depth_anything.infer_image(raw_image, args.input_size)
        depth = (depth - depth.min()) / (depth.max() - depth.min())

        output_path_image = os.path.join(args.outdir, os.path.splitext(os.path.basename(filename))[0] + '_depth' + '.png')
        
        if args.save_numpy:
            output_path_numpy = os.path.join(args.outdir, os.path.splitext(os.path.basename(filename))[0] + '_mde_depth.npy')
            np.save(output_path_numpy, depth)
        
        depth = depth * 255.0                               # 绿色为主的深度图
        depth = depth.astype(np.uint8)
        
        if args.grayscale:                                  # 灰度深度图
            depth = np.repeat(depth[..., np.newaxis], 3, axis=-1)
        else:
            depth = (cmap(depth)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)
        
        if args.pred_only:
            cv2.imwrite(output_path_image, depth)
        else:
            split_region = np.ones((raw_image.shape[0], 50, 3), dtype=np.uint8) * 255
            combined_result = cv2.hconcat([raw_image, split_region, depth])
            
            cv2.imwrite(output_path_image, combined_result)

        # depth_map = vis_depth(depth)                      # 蓝色为主的深度图
        # cv2.imwrite(output_path_image, depth_map)
```
Then run **DV2run.py** to obtain the depthmaps and depth.npy of the input images
```
python <.....>/FS4DGS/utils/DV2/metric_depth/DV2run.py --encoder vitl --load-from <.....>/FS4DGS/utils/DV2/metric_depth/checkpoints/depth_anything_v2_metric_hypersim_vitl.pth --max-depth 3.6 --img-path <location>/<scene>/images --outdir <location>/<scene>/depths --input-size 1000 --pred-only --save-numpy
 
```
## Key References
### Dataset
**Dynerf Dataset**: Neural 3D Video Synthesis from Multi-view Video, CVPR 2022. </br>
**Technicolor Dataset**: Dataset and Pipeline for Multi-View Light-Field Video, CVPR 2017 Workshop. </br>
**Google Immersive Dataset**: Immersive Light Field Video with a Layered Mesh Representation, SIGGRAPH 2020. </br>
**Meet Room Dataset**: Streaming Radiance Fields for 3D Video Synthesis, NeurIPS 2022. </br>

### Dynamic View Synthesis Using Gaussian Splatting
**RealTime4DGS**: Real-time Photorealistic Dynamic Scene Representation and Rendering with 4D Gaussian Splatting, ICLR 2024. </br>
**SpaceTimeGS**: Spacetime Gaussian Feature Splatting for Real-Time Dynamic View Synthesis, CVPR 2024. </br>
**E-D3DGS**: Per-Gaussian Embedding-Based Deformation for Deformable 3D Gaussian Splatting, ECCV 2024. </br>
**4DGS**: 4D Gaussian Splatting for Real-Time Dynamic Scene Rendering, CVPR 2024. </br>
**4D-Rotor-GS**: 4D-Rotor Gaussian Splatting: Towards Efficient Novel View Synthesis for Dynamic Scenes, SIGGRRAPH 2024.

### Few-shot Novel View Synthesis Using Gaussian Splatting
**FSGS**: FSGS: Real-Time Few-shot View Synthesis using Gaussian Splatting, ECCV 2024. </br>
**DNGS**: DNGaussian: Optimizing Sparse-View 3D Gaussian Radiance Fields with Global-Local Depth Normalization, CVPR 2024. </br>
**MVPGS**: MVPGS: Excavating Multi-view Priors for Gaussian Splatting from Sparse Input Views, ECCV 2024. </br>
**MVSGaussian**: MVSGaussian: Fast Generalizable Gaussian Splatting Reconstruction from Multi-View Stereo, ECCV 2024. </br>
**InstantSplat**: InstantSplat: Sparse-view SfM-free Gaussian Splatting in Seconds, arxiv. </br>
**GaussianObject**: GaussianObject: High-Quality 3D Object Reconstruction from Four Views with Gaussian Splatting, SIGGRAPH Asia 2024. </br>

### Learning-based Multi-view Stereo
**MVSFormer**: MVSFormer: Multi-View Stereo by Learning Robust Image Features and Temperature-based Depth, TMLR 2023. </br>
**MVSFormer++**: MVSFormer++: Revealing the Devil in Transformer’s Details for Multi-View Stereo, ICLR 2024. </br>
**DUSt3R**: DUSt3R: Geometric 3D Vision Made Easy, CVPR 2024. </br>
**MASt3R**: Grounding Image Matching in 3D with MASt3R,arxiv. </br>



