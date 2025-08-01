# Geometry-Consistent 4D Gaussian Splatting for Sparse-Input Dynamic View Synthesis

**This repository is the official implementation of "Geometry-Consistent 4D Gaussian Splatting for Sparse-Input Dynamic View Synthesis".**

## Get started

### Environment

The hardware and software requirements are the same as those of the [4D Gaussian Splatting](https://github.com/fudan-zvg/4d-gaussian-splatting), which this code is built upon. To setup the environment, please run the following command:

```shell
git clone https://github.com/liyw420/FS4DGS
cd GC-4DGS
conda env create --file environment.yml
conda activate gc4dgs
```

### Data preparation

**N3DV dataset:**

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

**Training:**
```
python train.py --config configs/<dataset>/<scene_name>_MVS.yaml
```
**Rendering:**
```
python render.py --model_path output/<dataset>/<scene_name>/ --loaded_pth output/<dataset>/<scene_name>/chkpnt_best.pth
```
**Evaluation:**
```
python metrics.py --model_path output/<dataset>/<scene_name>/
```
