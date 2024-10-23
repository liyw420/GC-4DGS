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
流程十分繁长，我不装了，直接中文。接下来得到训练集和测试集的相机信息：
```
export PYTHONPATH=<.....>/fs4dgs 
python scripts/pre_technicolor/imgs2poses.py --match_type exhaustive_matcher --scenedir <location>/<scene>
```






