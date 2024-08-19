from pyntcloud import PyntCloud
import pandas as pd
import numpy as np
import point_cloud_utils as pcu

from base_class import BaseClass


class Filter(BaseClass):
    def __init__(self, config_path):
        super().__init__(config_path)

    def downsample_voxel(self, filtered_lidar):
        filtered_lidar = pcu.downsample_point_cloud_on_voxel_grid(self.cfg.downsampling.voxel_size, filtered_lidar)
        return filtered_lidar

    def downsample_random(self, filtered_lidar, number=1000):
        size = filtered_lidar.shape[0]
        if size > number:
            idxs = np.random.choice(np.arange(size), number, replace=False)
            downsampled = filtered_lidar[idxs, :]
            return downsampled
        else:
            return filtered_lidar

    def downsample(self, lidar, method="both", random_points=1000):
        # Input lidar scan Nx3
        lidar = lidar.T
        if lidar.shape[0] > 1000:
            tmp1 = self.downsample_random(lidar[:, :3], random_points)
            tmp2 = self.downsample_voxel(lidar[:, :3])

            lidar = np.concatenate((tmp1, tmp2), axis=0)

        return lidar[:, :3]
