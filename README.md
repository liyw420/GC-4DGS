# GC-4DGS: Geometry-Consistent 4D Gaussian Splatting for Sparse-Input Dynamic View Synthesis

**This repository is the official implementation of "Geometry-Consistent 4D Gaussian Splatting for Sparse-Input Dynamic View Synthesis".** In this paper, we introduce Geometry-Consistent 4D Gaussian Splatting (GC-4DGS), a novel DVS framework to achieve real-time high-fidelity dynamic scene rendering from sparse input views.

## 🛠️ Pipeline
<div align="center">
  <img src="assets/pipeline.png"/>
</div><br/>

## 🌟 Get started

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

Download the [Neural 3D Video dataset](https://github.com/facebookresearch/Neural_3D_Video) and extract each scene to `./data/N3V`.

**Technicolor dataset:**

Please reach out to the authors of the paper [Dataset and Pipeline for Multi-View Light-Field Video](https://openaccess.thecvf.com/content_cvpr_2017_workshops/w27/papers/Sabater_Dataset_and_Pipeline_CVPR_2017_paper.pdf) for access to the Technicolor dataset. Please extract each scene to `./data/technicolor`. Our codebase expects the following directory structure for this dataset before preprocessing:
```
<location>
|---Fabien
|   |---Fabien_undist_<00257>_<08>.png
|   |---Fabien_undist_<.....>_<..>.png
|---Birthday
```
### Download Pretrained Models

**MVSFormer:**

Download the official pretrained [MVSFormer](https://github.com/ewrfcas/MVSFormer) weights (MVSFormer.zip and MVSFormer-Blended.zip) from the [official link](https://onedrive.live.com/?redeem=aHR0cHM6Ly8xZHJ2Lm1zL3UvcyFBaDJWa1VMbWtpcVByeUhfVGwyUFVTNklzODMxP2U9QmdDdU9Z&id=8F2A92E64291951D%216049&cid=8F2A92E64291951D). Extract the pretrained models to `./mvs_modules/pretrained/`.

**DepthAnythingV2:**

We use [Depth-Anything-V2-Large for metric depth estimation](https://github.com/DepthAnything/Depth-Anything-V2/tree/main/metric_depth), which is finetuned on indoor scenes (Hypersim). Clone the repository to `./utils/DV2`.

### Running

For data preprocessing, model training, rendering, and evaluation on N3DV Dataset:

```
bash ./scripts/pre_dynerf/run_train_eval.sh
```
For data preprocessing, model training, rendering, and evaluation on Technicolor Dataset:

```
bash ./scripts/pre_technicolor/run_train_eval.sh
```
## 🎥 Video Demos

To view the video demonstrations properly, please **download** them from the `./demos/` directory. Thanks for your understanding.
















