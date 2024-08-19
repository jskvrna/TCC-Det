import numpy as np
import torch

from base_class import BaseClass


class DataTransformer(BaseClass):
    def __init__(self, config_path):
        super().__init__(config_path)

    def compute_GT_angles_for_movingv2(self, batch_dict, correspondence, i):
        locations = batch_dict['locations'][i]
        moving = batch_dict['moving'][i]
        moving_angle = []

        for z in range(len(locations)):
            if len(locations[z]) > 0 and moving[z]:
                if len(locations[z]) < 3:
                    moving_angle.append(torch.inf)
                else:
                    ref_idx = None
                    for k in range(len(locations[z])):
                        if locations[z][k][3] == 0:
                            ref_idx = k
                        if 'flip_x' in batch_dict and batch_dict['flip_x'][i] > 0.5:
                            locations[z][k][1] = -1 * locations[z][k][1]
                        locations[z][k][:3] = torch.matmul(batch_dict['lidar_aug_matrix'][i, :3, :3].cpu(),
                                                              locations[z][k][:3].T).squeeze().T
                        locations[z][k][:3] += batch_dict['lidar_aug_matrix'][i, :3, 3].cpu()

                    estimation_arr = []

                    idx = ref_idx - 1
                    count = 0
                    while idx >= 0 and count < 5:
                        dist = np.sqrt((np.power(locations[z][ref_idx][0] - locations[z][idx][0], 2) +
                                np.power(locations[z][ref_idx][1] - locations[z][idx][1], 2)))
                        if dist > 1.5:
                            angle = np.arctan2(locations[z][ref_idx][1] - locations[z][idx][1],
                                               locations[z][ref_idx][0] - locations[z][idx][0])
                            estimation_arr.append(angle)
                            count += 1
                        idx -= 1

                    idx = ref_idx + 1
                    count = 0
                    while idx < len(locations[z]) and count < 5:
                        dist = np.sqrt((np.power(locations[z][idx][0] - locations[z][ref_idx][0], 2) +
                                        np.power(locations[z][idx][1] - locations[z][ref_idx][1], 2)))
                        if dist > 1.5:
                            angle = np.arctan2(locations[z][idx][1] - locations[z][ref_idx][1],
                                               locations[z][idx][0] - locations[z][ref_idx][0])
                            estimation_arr.append(angle)
                            count += 1
                        idx += 1

                    if len(estimation_arr) < 3:
                        moving_angle.append(torch.inf)
                    else:
                        if len(estimation_arr) % 2 == 0:
                            estimation_arr.append(estimation_arr[-1])
                        estimation_arr = np.array(estimation_arr)
                        predicted_angle = np.median(estimation_arr)
                        if predicted_angle > np.pi:
                            predicted_angle -= 2 * np.pi
                        moving_angle.append(predicted_angle)

            else:
                moving_angle.append(torch.inf)

        output = torch.zeros(len(correspondence), dtype=torch.float32, device='cuda')
        for z in range(len(correspondence)):
            output[z] = float(moving_angle[correspondence[z]])

        return output

    def transform_lidar_cam2_to_velo(self, lidar):
        #Input n x 3 lidar array
        padding = np.ones((lidar.shape[0], 1))

        lidar = np.concatenate((lidar, padding), axis=1)

        lidar[:, 3] = 1
        # Transform to the camera coordinate
        lidar = lidar.transpose()
        # This should be rectified already
        T_velo_to_cam2 = self.T_cam2_velo
        T_cam2_to_velo = np.linalg.inv(T_velo_to_cam2)
        lidar = np.matmul(T_cam2_to_velo, lidar)

        return lidar.T[:, :3]

    def transform_location_cam2_to_velo(self, locations):
        T_velo_to_cam2 = self.T_cam2_velo
        T_cam2_to_velo = np.linalg.inv(T_velo_to_cam2)

        for loc in locations:
            loc[:3] = np.matmul(T_cam2_to_velo[:3,:3], loc[:3])
            loc[:3] += T_cam2_to_velo[:3,3]

        return locations
