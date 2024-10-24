# FS4DGS: Few-shot Dynamic Novel View Synthesis using Gaussian Splatting

**This repository is the official implementation of "FS4DGS: Few-shot Dynamic Novel View Synthesis using Gaussian Splatting".** In this paper, we propose XXX.

## Get started

### Environment

The hardware and software requirements are the same as those of the [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting), which this code is built upon. To setup the environment, please run the following command:

```shell
git clone https://github.com/liyw420/FS4DGS
cd fs4dgs
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
export PYTHONPATH=<.....>/fs4dgs 
python scripts/pre_technicolor/imgs2poses.py --match_type exhaustive_matcher --scenedir <location>/<scene>
```
接下来得到训练集和测试集的相机信息(.json文件)：
```
python scripts/pre_technicolor/pose2MVS.py --path <location>/<scene>
```
然后利用MVSFormer得到点云以及深度数据和mask
```
python mvs2points.py --path <location>/<scene> --mvs_config <.....>/fs4dgs/mvs_modules/configs/config_mvsformer.json --dataset technicolor
```
基于open3d库对MVS得到的点云进行噪点过滤并降采样
```
python scripts/pre_technicolor/o3dPre.py ----mvs_input <location>/<scene>/mvs --downsampling_rate 0.1 --output_path <location>/<scene>
```
## Key References
### Dataset
**Dynerf Dataset**: Neural 3D Video Synthesis from Multi-view Video, CVPR 2022. </br>
**Technicolor Dataset**: Dataset and Pipeline for Multi-View Light-Field Video, CVPR 2017 Workshop. </br>
**Google Immersive Dataset**: Immersive Light Field Video with a Layered Mesh Representation, SIGGRAPH 2020. </br>
**Meet Room Dataset**: Streaming Radiance Fields for 3D Video Synthesis, Nips 2022. </br>

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



