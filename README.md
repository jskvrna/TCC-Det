# TCC-Det: Temporarily consistent cues for weakly-supervised 3D detection

**This is the official repository of the paper "TCC-Det: Temporarily consistent cues for weakly-supervised 3D detection" accepted at ECCV 2024.**

**Authors: [Jan Skvrna](https://jskvrna.github.io/), [Lukas Neumann](TODO)**

**Affiliation: [Visual Recognition Group at Czech Technical University in Prague](https://cyber.felk.cvut.cz/research/groups-teams/vrg/)**

Link to the paper: [ECCV2024](TODO)

![Intro Image](figures/intro_image.webp)

## Abstract

Accurate object detection in LiDAR point clouds is a key prerequisite of robust and safe autonomous driving and robotics applications. Training the 3D object detectors currently involves  the need to manually annotate vasts amounts of training data, which is very time-consuming and costly. As a result, the amount of annotated training data readily available is limited, and moreover these annotated datasets likely do not contain edge-case or otherwise rare instances, simply because the probability of them occurring in such a small dataset is low.

In this paper, we propose a method to train 3D object detector without any need for manual annotations, by exploiting existing off-the-shelf vision components and by using the consistency of the world around us. The method can therefore be used to train a 3D detector by only collecting sensor recordings in the real world, which is extremely cheap and allows training using orders of magnitude more data than traditional fully-supervised methods.

The method is evaluated on KITTI and Waymo Open datasets, where it outperforms all previous weakly-supervised methods and where it narrows the gap when compared to methods using human 3D labels.

For more details, please refer to the [paper](TODO).

## Description

Code is divided into two parts:
1. **Pseudo Ground Truth generator**: 