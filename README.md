# TCC-Det: Temporarily consistent cues for weakly-supervised 3D detection

**This is the official repository of the paper "TCC-Det: Temporarily consistent cues for weakly-supervised 3D detection" accepted at ECCV 2024.**

**Authors: [Jan Skvrna](https://jskvrna.github.io/), [Lukas Neumann](https://cmp.felk.cvut.cz/~neumalu1/)**

**Affiliation: [Visual Recognition Group at Czech Technical University in Prague](https://cyber.felk.cvut.cz/research/groups-teams/vrg/)**

Link to the paper: [ECCV2024](TODO)

![Intro Image](figures/intro_image.webp)

<figcaption>Figure 1: Combining raw unlabelled RGB camera and LiDAR sensor data across multiple frames in a temporally consistent manner allows us to exploit a generic off-the-shelf 2D object detector to train a 3D object (vehicle) detector for LiDAR point clouds.</figcaption>

---

## Abstract

Accurate object detection in LiDAR point clouds is a key prerequisite of robust and safe autonomous driving and robotics applications. Training the 3D object detectors currently involves  the need to manually annotate vasts amounts of training data, which is very time-consuming and costly. As a result, the amount of annotated training data readily available is limited, and moreover these annotated datasets likely do not contain edge-case or otherwise rare instances, simply because the probability of them occurring in such a small dataset is low.

In this paper, we propose a method to train 3D object detector without any need for manual annotations, by exploiting existing off-the-shelf vision components and by using the consistency of the world around us. The method can therefore be used to train a 3D detector by only collecting sensor recordings in the real world, which is extremely cheap and allows training using orders of magnitude more data than traditional fully-supervised methods.

The method is evaluated on KITTI and Waymo Open datasets, where it outperforms all previous weakly-supervised methods and where it narrows the gap when compared to methods using human 3D labels.

---

## Method

Code is divided into two main parts:
1. **Pseudo Ground Truth generator**: This part is further divided into multiple steps:
   1. (Waymo only) Decompressing the LiDAR point clouds from the ProtoBuf format. Action: *lidar_scans*
   2. Generating the precise frame-to-frame transformations. Action: *transformations*
   3. Running Mask-RCNN and tracker to obtain masks and correspondences. Action: *mask_tracking*
   4. Frames aggregation. Action: *frames_aggregation*
   5. Optimization of the aggregated frames to obtain precise pseudo ground truth. Action: *optimization*
2. **Training**:
   1. Using the pseudo ground truth to train the 3D object detector (OpenPCDet).
   2. Fine-tuning of the trained model on the pseudo ground truth with the additional losses (TFL and MAL).

For more details, please refer to the [paper](TODO).

![Method Image](figures/pipeline_weakly2.webp)

<figcaption>Figure 2: Training pipeline of the weakly-supervised 3D object detector relying on 2D detections and shape prior hypotheses.</figcaption>

---

## Installation

We recommend using the conda environment. Specifically the Python 3.10.14 version and CUDA 11.7.0 is recommended. Python 3.8 and 3.9 should work also fine.

If your machine is running Windows, please use the WSL2 with Ubuntu 22.04 LTS.

1. **Clone the repository:**
   ```bash
   git clone https://github.com/jskvrna/TCC-Det.git
2. **Install the requirements:**
   ```bash
   cd TCC-Det/
   pip install -r requirements.txt
3. **Build the [Pytorch3D](https://github.com/facebookresearch/pytorch3d/tree/main) library from source:**
   ```bash
   git clone https://github.com/facebookresearch/pytorch3d.git
   cd pytorch3d && pip install -e .
   ```
   If the build fails, because the cc1plus is killed, limit the number of jobs:
   ```bash
   cd pytorch3d && MAX_JOBS=2 pip install -e .
4. **Install the [Waymo Open Dataset](https://github.com/waymo-research/waymo-open-dataset) library:**
   ```bash
   pip install waymo-open-dataset-tf-2-11-0==1.6.1
   ```
   Unfortunately, there is some dependency issues within the packages, so ignore the warnings from pip.
3. **Install the [OpenPCDet](https://github.com/open-mmlab/OpenPCDet) library:**
   ```bash
   cd modified_openpcdet
   python setup.py develop
   ```
   Again, if the build is killed, limit the number of jobs:
   ```bash
   MAX_JOBS=2 python setup.py develop
---

## Citation
TODO