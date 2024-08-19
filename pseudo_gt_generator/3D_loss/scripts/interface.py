import copy
import numpy as np
import torch
from pytorch3d.transforms import euler_angles_to_matrix
import torch.nn.functional as F
from base_class import BaseClass


class Interface(BaseClass):
    def __init__(self, config_path):
        super().__init__(config_path)

    def load_additional_data(self, data_dict, frame_idx):
        return self.load_frame_data(data_dict, frame_idx)

    def get_lidar_to_fit(self, data_dict, sample_number):
        lidars = data_dict['lidars'][sample_number]

        for i in range(len(lidars)):
            lidars[i] = F.pad(lidars[i].cuda(), (0, 1), 'constant', 1.)
            if 'flip_x' in data_dict and data_dict['flip_x'][sample_number] > 0.5:
                lidars[i][:, 1] = -1 * lidars[i][:, 1]
            lidars[i] = torch.matmul(data_dict['lidar_aug_matrix'][sample_number], lidars[i].T).squeeze()[:3, :].T

        return lidars

    def get_lidar_to_fit_waymo(self, batch_dict, batch):
        to_return_standing = batch_dict['standing'][batch]
        for i in range(len(to_return_standing)):
            to_return_standing[i] = torch.tensor(to_return_standing[i], device=torch.device('cuda'), dtype=torch.float32)
        to_return_moving = []
        for i in range(len(batch_dict['moving'][batch][0])):
            to_return_moving.append(
                torch.tensor(batch_dict['moving'][batch][0][i][2], device=torch.device('cuda'), dtype=torch.float32).T[:,:3])

        to_return = to_return_standing + to_return_moving

        for i in range(len(to_return)):
            to_return[i] = F.pad(to_return[i], (0, 1), 'constant', 1.)
            if 'flip_x' in batch_dict and batch_dict['flip_x'][batch] > 0.5:
                to_return[i][:, 1] = -1 * to_return[i][:, 1]
            if 'flip_y' in batch_dict and batch_dict['flip_y'][batch] > 0.5:
                to_return[i][:, 0] = -1 * to_return[i][:, 0]
            to_return[i] = torch.matmul(batch_dict['lidar_aug_matrix'][batch], to_return[i].T).squeeze()[:3, :].T

        return to_return

    def get_templates_from_anchors(self, bboxes, anchors, moving, moving_cars_angles):
        centers = bboxes[:, :3]
        extents = bboxes[:, 3:6]
        angles = bboxes[:, 6]
        output = torch.zeros((bboxes.shape[0], 2, self.lidar_car_template_non_filt[0].shape[0],
                           self.lidar_car_template_non_filt[0].shape[1]), dtype=torch.float32).cuda()

        for z in range(bboxes.shape[0]):
            tmp_template = self.lidar_car_template_non_filt[0].detach().clone()
            # First
            x_temp_extent = self.cfg.templates.template_length
            y_temp_extent = self.cfg.templates.template_width
            z_temp_extent = self.cfg.templates.template_height

            scale_x = extents[z][0].detach() / x_temp_extent
            scale_y = extents[z][1].detach() / y_temp_extent
            scale_z = extents[z][2].detach() / z_temp_extent

            tmp_template[:, 0] *= scale_x
            tmp_template[:, 1] *= scale_y
            tmp_template[:, 2] *= scale_z

            tmp_template2 = tmp_template.detach().clone()

            # Second rotate
            if not moving[z] or moving_cars_angles[z] == torch.inf or self.cfg.optimization.use_rot_gradient_for_moving: #It is a standing car
                if anchors[z, 6] > 0.1: #If it is the rotated anchor
                    r = euler_angles_to_matrix(torch.tensor([0., 0., 1.57], dtype=torch.float32), "XYZ").cuda()
                    tmp_template = torch.matmul(r, tmp_template.T).T
                    r = euler_angles_to_matrix(torch.tensor([0., 0., 4.71], dtype=torch.float32), "XYZ").cuda()
                    tmp_template2 = torch.matmul(r, tmp_template2.T).T
                    rot_angle = angles[z] - 1.57
                    r = torch.eye(3, dtype=torch.float32, device='cuda')
                    r[0, 0] = torch.cos(rot_angle)
                    r[0, 1] = -torch.sin(rot_angle)
                    r[1, 0] = torch.sin(rot_angle)
                    r[1, 1] = torch.cos(rot_angle)
                    output[z, 0] = torch.matmul(r, tmp_template.T).T
                    output[z, 1] = torch.matmul(r, tmp_template2.T).T
                else:
                    r = euler_angles_to_matrix(torch.tensor([0., 0., 3.14], dtype=torch.float32), "XYZ").cuda()
                    tmp_template2 = torch.matmul(r, tmp_template2.T).T
                    rot_angle = angles[z]
                    r = torch.eye(3, dtype=torch.float32, device='cuda')
                    r[0, 0] = torch.cos(rot_angle)
                    r[0, 1] = -torch.sin(rot_angle)
                    r[1, 0] = torch.sin(rot_angle)
                    r[1, 1] = torch.cos(rot_angle)
                    output[z, 0] = torch.matmul(r, tmp_template.T).T
                    output[z, 1] = torch.matmul(r, tmp_template2.T).T
            else:  # We know the angle.
                r = euler_angles_to_matrix(torch.tensor([0., 0., moving_cars_angles[z]], dtype=torch.float32), "XYZ").cuda()
                output[z, 0] = torch.matmul(r, output[z, 0].T).T
                output[z, 1] = torch.matmul(r, output[z, 1].T).T

            # Third translate
            output[z, 0][:, 0] = output[z, 0][:, 0] + centers[z][0]
            output[z, 0][:, 1] = output[z, 0][:, 1] + centers[z][1]
            output[z, 0][:, 2] = output[z, 0][:, 2] + centers[z][2]

            output[z, 1][:, 0] = output[z, 1][:, 0] + centers[z][0]
            output[z, 1][:, 1] = output[z, 1][:, 1] + centers[z][1]
            output[z, 1][:, 2] = output[z, 1][:, 2] + centers[z][2]

        return output

    def get_standing_moving_info(self, batch_dict, correspondence, batch_id):
        moving = batch_dict['moving'][batch_id]
        output = torch.zeros(len(correspondence), dtype=torch.bool, device='cuda')
        #0 standing, 1 moving
        for i in range(len(correspondence)):
            if moving[correspondence[i]]:
                output[i] = True
        return output

    def get_target_car_mask(self, corresponding_car_idx, batch_dict, batch_idx):
        return self.create_target_car_mask(corresponding_car_idx, batch_dict, batch_idx)

    def render_bbox_to_mask_single(self, bbox, data_dict, frame_idx, batch_idx, sharp=False, template_idx=0):
        return self.render_by_single_bbox(bbox, data_dict, frame_idx, batch_idx, sharp, template_idx)

    def render_bbox_to_mask_single_distloss(self, bbox, data_dict, frame_idx, batch_idx, visu=False):
        return self.render_by_single_bbox_distloss(bbox, data_dict, frame_idx, batch_idx, visu)

    def vizualize_lidar(self, data_dict):
        self.vizu_lidar(data_dict)

    def vizualize_show(self, visu_type):
        self.vizu_show(visu_type)

    def vizualize_gt_bboxes(self, data_dict):
        self.vizu_gt_bbox(data_dict)

    def vizualize_det_bboxes(self, det_bbox):
        self.vizu_det_bbox(det_bbox)

    def vizualize_lidar_to_fit(self, det_bbox, colors=None):
        self.vizu_lidar_to_fit(det_bbox, colors)

    def vizualize_templates(self, templates, direction, colors=None):
        self.vizu_templates(templates, direction, colors)

    def vizualize_orig_gt_bboxes(self, det_bbox):
        self.vizu_orig_gt_bbox(det_bbox)

    def vizualize_gradient(self, loc, grad):
        self.vizu_gradient(loc, grad)

    def vizualize_image_masks(self, frame_idx, mask_orig, mask_loss=None):
        self.vizu_image_masks(frame_idx, mask_orig, mask_loss)






