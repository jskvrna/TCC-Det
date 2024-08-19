# Modified by: Jan Skvrna for the purpose of the TCC-Det
# Modified parts are marked with the comment: # Start TCC-Det and # End TCC-Det

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# Start TCC-Det
import cv2
import matplotlib.pyplot as plt
# End TCC-Det

from ...utils import box_coder_utils, common_utils, loss_utils
from ..model_utils.model_nms_utils import class_agnostic_nms
from .target_assigner.proposal_target_layer import ProposalTargetLayer


class RoIHeadTemplate(nn.Module):
    # Start TCC-Det
    def __init__(self, num_class, model_cfg, dataset, **kwargs):
        super().__init__()
        self.dataset = dataset
        # End TCC-Det
        self.model_cfg = model_cfg
        self.num_class = num_class
        self.box_coder = getattr(box_coder_utils, self.model_cfg.TARGET_CONFIG.BOX_CODER)(
            **self.model_cfg.TARGET_CONFIG.get('BOX_CODER_CONFIG', {})
        )
        self.proposal_target_layer = ProposalTargetLayer(roi_sampler_cfg=self.model_cfg.TARGET_CONFIG)
        self.build_losses(self.model_cfg.LOSS_CONFIG)
        self.forward_ret_dict = None
        # Start TCC-Det
        self.use_ROI_head = self.model_cfg.get('USE_ROI_HEAD', True)
        self.use_template_loss = self.model_cfg.get('USE_TEMPLATE_LOSS', False)
        self.mask_loss_div = self.model_cfg.get('MASK_LOSS_DIV', 33.)
        self.templ_loss_div = self.model_cfg.get('TEMPL_LOSS_DIV', 1.)
        self.use_L1Loss = self.model_cfg.get('LOSS_L1', False)
        self.use_L2Loss = self.model_cfg.get('LOSS_L2', False)
        self.use_SmoothL1Loss = self.model_cfg.get('LOSS_SMOOTHL1', True)
        self.use_silhouette = self.model_cfg.get('USE_SILHOUETTE', True)
        self.use_bin_loss = self.model_cfg.get('USE_BIN_LOSS', False)
        self.use_mask_loss = self.model_cfg.get('USE_MASK_LOSS', False)
        self.use_orig_loss = self.model_cfg.get('USE_ORIG_LOSS', True)
        self.use_chamfer_loss =self.model_cfg.get('USE_CHAMFER_LOSS', False)
        self.FN_epsilon = self.model_cfg.get('FN_EPSILON', 0.001)

        # Visualization settings.
        self.print_gradients = False
        self.visu_lidar = False
        self.visu_masks = False
        self.visu_type = "step"  # Possible: animation, step
        self.grad_visu_type = "all"  # Possible: all, step

        self.smoothL1 = nn.SmoothL1Loss()

        np.set_printoptions(formatter={'float': lambda x: "{0:0.4f}".format(x)})

        self.visu_batch_box = None
        self.visu_scores = None
        self.point_clouds_to_fit = None
        self.templates = None
        self.direction_templates = None
        self.rois = None
        self.batch_dict = None
        self.mask = None
        self.mask_target = None
        self.correspondence = None
        # End TCC-Det


    def build_losses(self, losses_cfg):
        self.add_module(
            'reg_loss_func',
            loss_utils.WeightedSmoothL1Loss(code_weights=losses_cfg.LOSS_WEIGHTS['code_weights'])
        )

    def make_fc_layers(self, input_channels, output_channels, fc_list):
        fc_layers = []
        pre_channel = input_channels
        for k in range(0, fc_list.__len__()):
            fc_layers.extend([
                nn.Conv1d(pre_channel, fc_list[k], kernel_size=1, bias=False),
                nn.BatchNorm1d(fc_list[k]),
                nn.ReLU()
            ])
            pre_channel = fc_list[k]
            if self.model_cfg.DP_RATIO >= 0 and k == 0:
                fc_layers.append(nn.Dropout(self.model_cfg.DP_RATIO))
        fc_layers.append(nn.Conv1d(pre_channel, output_channels, kernel_size=1, bias=True))
        fc_layers = nn.Sequential(*fc_layers)
        return fc_layers

    @torch.no_grad()
    def proposal_layer(self, batch_dict, nms_config):
        """
        Args:
            batch_dict:
                batch_size:
                batch_cls_preds: (B, num_boxes, num_classes | 1) or (N1+N2+..., num_classes | 1)
                batch_box_preds: (B, num_boxes, 7+C) or (N1+N2+..., 7+C)
                cls_preds_normalized: indicate whether batch_cls_preds is normalized
                batch_index: optional (N1+N2+...)
            nms_config:

        Returns:
            batch_dict:
                rois: (B, num_rois, 7+C)
                roi_scores: (B, num_rois)
                roi_labels: (B, num_rois)

        """
        if batch_dict.get('rois', None) is not None:
            return batch_dict
            
        batch_size = batch_dict['batch_size']
        batch_box_preds = batch_dict['batch_box_preds']
        batch_cls_preds = batch_dict['batch_cls_preds']
        rois = batch_box_preds.new_zeros((batch_size, nms_config.NMS_POST_MAXSIZE, batch_box_preds.shape[-1]))
        roi_scores = batch_box_preds.new_zeros((batch_size, nms_config.NMS_POST_MAXSIZE))
        roi_labels = batch_box_preds.new_zeros((batch_size, nms_config.NMS_POST_MAXSIZE), dtype=torch.long)

        for index in range(batch_size):
            if batch_dict.get('batch_index', None) is not None:
                assert batch_cls_preds.shape.__len__() == 2
                batch_mask = (batch_dict['batch_index'] == index)
            else:
                assert batch_dict['batch_cls_preds'].shape.__len__() == 3
                batch_mask = index
            box_preds = batch_box_preds[batch_mask]
            cls_preds = batch_cls_preds[batch_mask]

            cur_roi_scores, cur_roi_labels = torch.max(cls_preds, dim=1)

            if nms_config.MULTI_CLASSES_NMS:
                raise NotImplementedError
            else:
                selected, selected_scores = class_agnostic_nms(
                    box_scores=cur_roi_scores, box_preds=box_preds, nms_config=nms_config
                )

            rois[index, :len(selected), :] = box_preds[selected]
            roi_scores[index, :len(selected)] = cur_roi_scores[selected]
            roi_labels[index, :len(selected)] = cur_roi_labels[selected]

        batch_dict['rois'] = rois
        batch_dict['roi_scores'] = roi_scores
        batch_dict['roi_labels'] = roi_labels + 1
        batch_dict['has_class_labels'] = True if batch_cls_preds.shape[-1] > 1 else False
        batch_dict.pop('batch_index', None)
        return batch_dict

    def assign_targets(self, batch_dict):
        batch_size = batch_dict['batch_size']
        with torch.no_grad():
            targets_dict = self.proposal_target_layer.forward(batch_dict)

        rois = targets_dict['rois']  # (B, N, 7 + C)
        gt_of_rois = targets_dict['gt_of_rois']  # (B, N, 7 + C + 1)
        targets_dict['gt_of_rois_src'] = gt_of_rois.clone().detach()

        # canonical transformation
        roi_center = rois[:, :, 0:3]
        roi_ry = rois[:, :, 6] % (2 * np.pi)
        gt_of_rois[:, :, 0:3] = gt_of_rois[:, :, 0:3] - roi_center
        gt_of_rois[:, :, 6] = gt_of_rois[:, :, 6] - roi_ry

        # transfer LiDAR coords to local coords
        gt_of_rois = common_utils.rotate_points_along_z(
            points=gt_of_rois.view(-1, 1, gt_of_rois.shape[-1]), angle=-roi_ry.view(-1)
        ).view(batch_size, -1, gt_of_rois.shape[-1])

        # flip orientation if rois have opposite orientation
        heading_label = gt_of_rois[:, :, 6] % (2 * np.pi)  # 0 ~ 2pi
        opposite_flag = (heading_label > np.pi * 0.5) & (heading_label < np.pi * 1.5)
        heading_label[opposite_flag] = (heading_label[opposite_flag] + np.pi) % (2 * np.pi)  # (0 ~ pi/2, 3pi/2 ~ 2pi)
        flag = heading_label > np.pi
        heading_label[flag] = heading_label[flag] - np.pi * 2  # (-pi/2, pi/2)
        heading_label = torch.clamp(heading_label, min=-np.pi / 2, max=np.pi / 2)

        gt_of_rois[:, :, 6] = heading_label
        targets_dict['gt_of_rois'] = gt_of_rois
        return targets_dict

    def get_box_reg_layer_loss(self, forward_ret_dict):
        loss_cfgs = self.model_cfg.LOSS_CONFIG
        code_size = self.box_coder.code_size
        reg_valid_mask = forward_ret_dict['reg_valid_mask'].view(-1)
        gt_boxes3d_ct = forward_ret_dict['gt_of_rois'][..., 0:code_size]
        gt_of_rois_src = forward_ret_dict['gt_of_rois_src'][..., 0:code_size].view(-1, code_size)
        rcnn_reg = forward_ret_dict['rcnn_reg']  # (rcnn_batch_size, C)
        roi_boxes3d = forward_ret_dict['rois']
        rcnn_batch_size = gt_boxes3d_ct.view(-1, code_size).shape[0]

        # Start TCC-Det
        #rcnn_reg = rcnn_reg.clone()
        #rcnn_reg[:, 0] = rcnn_reg[:, 0].detach()
        #rcnn_reg[:, 1] = rcnn_reg[:, 1].detach()
        #rcnn_reg[:, 2] = rcnn_reg[:, 2].detach()
        #rcnn_reg[:, 5] = rcnn_reg[:, 5].detach()
        #rcnn_reg[:, 6] = rcnn_reg[:, 6].detach()
        # End TCC-Det

        fg_mask = (reg_valid_mask > 0)
        fg_sum = fg_mask.long().sum().item()

        tb_dict = {}

        if loss_cfgs.REG_LOSS == 'smooth-l1':
            rois_anchor = roi_boxes3d.clone().detach().view(-1, code_size)
            rois_anchor[:, 0:3] = 0
            rois_anchor[:, 6] = 0
            reg_targets = self.box_coder.encode_torch(
                gt_boxes3d_ct.view(rcnn_batch_size, code_size), rois_anchor
            )

            rcnn_loss_reg = self.reg_loss_func(
                rcnn_reg.view(rcnn_batch_size, -1).unsqueeze(dim=0),
                reg_targets.unsqueeze(dim=0),
            )  # [B, M, 7]
            rcnn_loss_reg = (rcnn_loss_reg.view(rcnn_batch_size, -1) * fg_mask.unsqueeze(dim=-1).float()).sum() / max(fg_sum, 1)
            rcnn_loss_reg = rcnn_loss_reg * loss_cfgs.LOSS_WEIGHTS['rcnn_reg_weight']
            tb_dict['rcnn_loss_reg'] = rcnn_loss_reg.item()

            if loss_cfgs.CORNER_LOSS_REGULARIZATION and fg_sum > 0:
                # TODO: NEED to BE CHECK
                fg_rcnn_reg = rcnn_reg.view(rcnn_batch_size, -1)[fg_mask]
                fg_roi_boxes3d = roi_boxes3d.view(-1, code_size)[fg_mask]

                fg_roi_boxes3d = fg_roi_boxes3d.view(1, -1, code_size)
                batch_anchors = fg_roi_boxes3d.clone().detach()
                roi_ry = fg_roi_boxes3d[:, :, 6].view(-1)
                roi_xyz = fg_roi_boxes3d[:, :, 0:3].view(-1, 3)
                batch_anchors[:, :, 0:3] = 0
                rcnn_boxes3d = self.box_coder.decode_torch(
                    fg_rcnn_reg.view(batch_anchors.shape[0], -1, code_size), batch_anchors
                ).view(-1, code_size)

                rcnn_boxes3d = common_utils.rotate_points_along_z(
                    rcnn_boxes3d.unsqueeze(dim=1), roi_ry
                ).squeeze(dim=1)
                rcnn_boxes3d[:, 0:3] += roi_xyz

                loss_corner = loss_utils.get_corner_loss_lidar(
                    rcnn_boxes3d[:, 0:7],
                    gt_of_rois_src[fg_mask][:, 0:7]
                )
                loss_corner = loss_corner.mean()
                loss_corner = loss_corner * loss_cfgs.LOSS_WEIGHTS['rcnn_corner_weight']

                rcnn_loss_reg += loss_corner
                tb_dict['rcnn_loss_corner'] = loss_corner.item()
        else:
            raise NotImplementedError
        # Start TCC-Det
        if self.visu_lidar:
            reg_valid_mask = forward_ret_dict['reg_valid_mask']
            batch_box_preds = forward_ret_dict['batch_box_preds_RCNN']
            batch_box_scores = forward_ret_dict['batch_cls_preds_RCNN']
            batch_size = int(batch_box_preds.shape[0])

            for i in range(batch_size):
                if self.print_gradients:
                    if self.batch_dict['dataset'] == 'waymo':
                        print(self.batch_dict['frame_id'][i])
                    else:
                        print(int(self.batch_dict['frame_id'][i]))
                positives_idx = torch.nonzero(reg_valid_mask[i])[:, 0]

                batch_box_preds_pos = batch_box_preds[i, :, :]

                self.visu_batch_box = batch_box_preds_pos.detach().clone().cpu()
                self.custom_loader.vizualize_lidar(self.batch_dict)
                self.custom_loader.vizualize_gt_bboxes(self.batch_dict)
                self.custom_loader.vizualize_det_bboxes(self.visu_batch_box)
                self.custom_loader.vizualize_show(self.visu_type)
        # End TCC-Det
        return rcnn_loss_reg, tb_dict
    # Start TCC-Det
    def get_box_reg_layer_loss_custom(self, forward_ret_dict):
        reg_valid_mask = forward_ret_dict['reg_valid_mask']
        batch_box_preds = forward_ret_dict['batch_box_preds_RCNN']
        batch_box_scores = forward_ret_dict['batch_cls_preds_RCNN']
        batch_size = int(batch_box_preds.shape[0])
        rois = forward_ret_dict['rois']
        gt_boxes = forward_ret_dict['gt_of_rois_src']

        loss = torch.tensor(0.).cuda()

        for i in range(batch_size):
            if self.print_gradients:
                if self.batch_dict['dataset'] == 'waymo':
                    print(self.batch_dict['frame_id'][i])
                else:
                    print(int(self.batch_dict['frame_id'][i]))
            positives_idx = torch.nonzero(reg_valid_mask[i])[:, 0]

            batch_box_preds_pos = batch_box_preds[i, positives_idx[:], :]
            batch_cls_preds_pos = batch_box_scores[i, positives_idx[:], :]

            reg_weights = torch.ones(batch_box_preds_pos.shape[0]).cuda()
            pos_normalizer = reg_valid_mask[i].sum(0, keepdim=True).float()
            reg_weights /= torch.clamp(pos_normalizer, min=1.0)

            if self.print_gradients:
                batch_box_preds_pos.register_hook(self.print_grad)

            point_clouds = self.dataset.custom_loader.get_lidar_to_fit(self.batch_dict, i)

            pcloud_centers = torch.zeros((len(point_clouds), 3), device='cuda')
            for z in range(len(point_clouds)):
                pcloud_centers[z, :] = torch.median(point_clouds[z], dim=0)[0]

            if batch_box_preds_pos.shape[0] == 0 or len(point_clouds) == 0:
                continue
            distances = torch.cdist(batch_box_preds_pos[:, 0:3], pcloud_centers)
            correspondence = torch.argmin(distances, dim=1)
            self.correspondence = correspondence

            moving_cars_angles = self.dataset.custom_loader.compute_GT_angles_for_movingv2(self.batch_dict, correspondence, i)

            standing_moving_info = self.dataset.custom_loader.get_standing_moving_info(self.batch_dict, correspondence, i)

            # Now lets generate template for each bbox prediction
            templates = self.dataset.custom_loader.get_templates_from_anchors(batch_box_preds_pos, torch.zeros_like(batch_box_preds_pos, device='cuda', dtype=torch.float32),
                                                                      standing_moving_info, moving_cars_angles)

            tmp_loss = torch.tensor(0.).cuda()

            template_loss, direction_templates = self.compute_template_loss(templates, point_clouds, correspondence,
                                                                            standing_moving_info, moving_cars_angles,
                                                                            batch_box_preds_pos)

            if self.use_mask_loss:
                if self.batch_dict['dataset'] == 'waymo':
                    mask_loss = self.compute_mask_loss_all(None, batch_box_preds_pos,
                                                           correspondence, i)
                else:
                    mask_loss = self.compute_mask_loss_all(int(self.batch_dict['frame_id'][i]), batch_box_preds_pos,
                                                           correspondence, i)
            else:
                mask_loss = 0.

            print("Template loss: ", template_loss, "     Mask loss: ", mask_loss / self.mask_loss_div, "     Frame idx: ", self.batch_dict['frame_id'][i])

            if self.use_bin_loss:
                tmp_loss = tmp_loss + template_loss / self.templ_loss_div
            if self.use_mask_loss:
                tmp_loss = tmp_loss + mask_loss / self.mask_loss_div
            loss += tmp_loss

            if self.print_gradients and self.visu_lidar:
                print(self.visu_batch_box)
                self.visu_batch_box = batch_box_preds_pos.detach().clone().cpu()
                self.visu_scores = batch_cls_preds_pos.detach().clone().cpu()
                self.point_clouds_to_fit = point_clouds
                self.templates = templates
                self.direction_templates = direction_templates
                self.rois = rois[i, positives_idx[:], :]

        loss = loss / batch_size
        tb_dict = {'rcnn_loss_reg': loss.item()}

        return loss, tb_dict

    def compute_template_loss(self, templates, point_clouds, correspondence, standing_moving_info, moving_cars_angles, batch_box_preds_pos):
        tmp_loss = torch.tensor(0.).cuda()
        direction_templates = torch.zeros(batch_box_preds_pos.shape[0], dtype=torch.int8)
        for z in range(templates.shape[0]):
            tmp_loss_front = torch.tensor(0.).cuda()
            tmp_loss_back = torch.tensor(0.).cuda()

            distances1 = torch.cdist(point_clouds[correspondence[z]], templates[z, 0])
            distances2 = torch.cdist(point_clouds[correspondence[z]], templates[z, 1])

            closest_dist_temp_to_scan1, _ = torch.min(distances1, dim=0)
            closest_dist_scan_to_temp1, _ = torch.min(distances1, dim=1)

            closest_dist_temp_to_scan2, _ = torch.min(distances2, dim=0)
            closest_dist_scan_to_temp2, _ = torch.min(distances2, dim=1)

            if not self.use_chamfer_loss:
                closest_dist_temp_to_scan1 = torch.sigmoid(10. * closest_dist_temp_to_scan1) - 0.5
                closest_dist_scan_to_temp1 = torch.sigmoid(10. * closest_dist_scan_to_temp1) - 0.5
                closest_dist_temp_to_scan2 = torch.sigmoid(10. * closest_dist_temp_to_scan2) - 0.5
                closest_dist_scan_to_temp2 = torch.sigmoid(10. * closest_dist_scan_to_temp2) - 0.5

            tmp_loss_front += torch.sum(closest_dist_temp_to_scan1) / templates[z, 0].shape[0]
            tmp_loss_front += torch.sum(closest_dist_scan_to_temp1) / point_clouds[correspondence[z]].shape[0]
            tmp_loss_back += torch.sum(closest_dist_temp_to_scan2) / templates[z, 1].shape[0]
            tmp_loss_back += torch.sum(closest_dist_scan_to_temp2) / point_clouds[correspondence[z]].shape[0]

            tmp_loss_front = tmp_loss_front
            tmp_loss_back = tmp_loss_back

            if standing_moving_info[z] == 1 and moving_cars_angles[z] != torch.inf:
                rad_pred_encoding = torch.sin(batch_box_preds_pos[z, 6]) * torch.cos(moving_cars_angles[z])
                rad_tg_encoding = torch.cos(batch_box_preds_pos[z, 6]) * torch.sin(moving_cars_angles[z])

                loss_sin = self.smoothL1(rad_pred_encoding, rad_tg_encoding)
                if self.visu_lidar:
                    pass
                    # print("Angle loss: ", loss_sin.detach().cpu().numpy())
                tmp_loss_front += loss_sin
                tmp_loss_back += loss_sin

            if self.visu_lidar:
                pass
                # print(batch_box_preds_pos[z].detach().cpu().numpy())
            if tmp_loss_front.item() > tmp_loss_back.item():
                tmp_loss += tmp_loss_back
                if self.visu_lidar:
                    direction_templates[z] = 1
                    # print(tmp_loss_back.detach().cpu().numpy(),
                    #      loss_gt_boxes[correspondence_to_gt[z]].detach().cpu().numpy())
            else:
                tmp_loss += tmp_loss_front
                if self.visu_lidar:
                    direction_templates[z] = 0
                    # print(tmp_loss_front.detach().cpu().numpy(),
                    #      loss_gt_boxes[correspondence_to_gt[z]].detach().cpu().numpy())
        tmp_loss = tmp_loss / templates.shape[0]

        return tmp_loss, direction_templates

    def compute_mask_loss_all(self, frame_idx, bboxes, correspondence, batch_idx):
        bboxes_rot = bboxes.clone()
        bboxes_rot[:, 6] = bboxes_rot[:, 6] + torch.pi

        if self.batch_dict['dataset'] == 'waymo':
            self.dataset.custom_loader.init_rendering_waymo(self.batch_dict, batch_idx)
        else:
            self.dataset.custom_loader.init_rendering(self.batch_dict, batch_idx)

        loss = torch.tensor(0.).cuda()

        for i in range(bboxes.shape[0]):
            if self.use_silhouette:
                # Decide in which camera is the car
                if self.batch_dict['dataset'] == 'waymo':
                    mask_target, mask_target_weights = self.dataset.custom_loader.create_target_car_mask_waymo(correspondence[i],
                                                                                              self.batch_dict, batch_idx)
                    if mask_target is None:
                        continue
                else:
                    mask_target, mask_target_weights = self.dataset.custom_loader.get_target_car_mask(correspondence[i],
                                                                                              self.batch_dict,
                                                                                              batch_idx)
                if self.batch_dict['dataset'] == 'waymo':
                    mask_bbox, mask_bbox_visu, camera_idx = self.dataset.custom_loader.get_target_camera_index(self.batch_dict, bboxes[i], mask_target, batch_idx)
                    if camera_idx == -1:
                        continue
                    mask_bbox_rot, mask_bbox_visu_rot = self.dataset.custom_loader.render_by_single_bbox_waymo(bboxes_rot[i], self.batch_dict, frame_idx,batch_idx, False, False, camera_idx)
                    mask_bbox_passat, mask_bbox_visu_passat = self.dataset.custom_loader.render_by_single_bbox_waymo(bboxes[i], self.batch_dict, frame_idx,batch_idx, False, False, camera_idx)
                    mask_bbox_rot_passat, mask_bbox_visu_rot_passat = self.dataset.custom_loader.render_by_single_bbox_waymo(bboxes_rot[i], self.batch_dict, frame_idx, batch_idx, False, True, camera_idx)
                else:
                    mask_bbox, _ = self.dataset.custom_loader.render_bbox_to_mask_single(bboxes[i], self.batch_dict, frame_idx, batch_idx, False, 0)
                    mask_bbox_rot, _ = self.dataset.custom_loader.render_bbox_to_mask_single(bboxes_rot[i], self.batch_dict, frame_idx, batch_idx, False, 0)
                    mask_bbox_passat, _ = self.dataset.custom_loader.render_bbox_to_mask_single(bboxes[i], self.batch_dict,frame_idx, batch_idx, False, 1)
                    mask_bbox_rot_passat, _ = self.dataset.custom_loader.render_bbox_to_mask_single(bboxes_rot[i], self.batch_dict, frame_idx, batch_idx, False, 1)
                    mask_bbox_suv, _ = self.dataset.custom_loader.render_bbox_to_mask_single( bboxes[i], self.batch_dict, frame_idx, batch_idx, False, 3)
                    mask_bbox_rot_suv, _ = self.dataset.custom_loader.render_bbox_to_mask_single(bboxes_rot[i], self.batch_dict, frame_idx, batch_idx, False, 3)
                    mask_bbox_mpv, _ = self.dataset.custom_loader.render_bbox_to_mask_single(bboxes[i], self.batch_dict,frame_idx, batch_idx,False, 2)
                    mask_bbox_rot_mpv, _ = self.dataset.custom_loader.render_bbox_to_mask_single(bboxes_rot[i], self.batch_dict, frame_idx, batch_idx, False, 2)


                loss_function = nn.BCELoss(weight=mask_target_weights, reduction='sum')

                mask_bbox = mask_bbox.squeeze(3).squeeze(0)
                mask_bbox_rot = mask_bbox_rot.squeeze(3).squeeze(0)
                mask_bbox_passat = mask_bbox_passat.squeeze(3).squeeze(0)
                mask_bbox_rot_passat = mask_bbox_rot_passat.squeeze(3).squeeze(0)
                mask_bbox_suv = mask_bbox_suv.squeeze(3).squeeze(0)
                mask_bbox_rot_suv = mask_bbox_rot_suv.squeeze(3).squeeze(0)
                mask_bbox_mpv = mask_bbox_mpv.squeeze(3).squeeze(0)
                mask_bbox_rot_mpv = mask_bbox_rot_mpv.squeeze(3).squeeze(0)

                pixel_FN = torch.logical_and(mask_bbox < self.FN_epsilon, mask_target > 0.5)
                pixel_FN_rot = torch.logical_and(mask_bbox_rot < self.FN_epsilon, mask_target > 0.5)
                pixel_FN_passat = torch.logical_and(mask_bbox_passat < self.FN_epsilon, mask_target > 0.5)
                pixel_FN_rot_passat = torch.logical_and(mask_bbox_rot_passat < self.FN_epsilon, mask_target > 0.5)
                pixel_FN_suv = torch.logical_and(mask_bbox_suv < self.FN_epsilon, mask_target > 0.5)
                pixel_FN_rot_suv = torch.logical_and(mask_bbox_rot_suv < self.FN_epsilon, mask_target > 0.5)
                pixel_FN_mpv = torch.logical_and(mask_bbox_mpv < self.FN_epsilon, mask_target > 0.5)
                pixel_FN_rot_mpv = torch.logical_and(mask_bbox_rot_mpv < self.FN_epsilon, mask_target > 0.5)

                mask_bbox[pixel_FN] = mask_bbox[pixel_FN] + self.FN_epsilon
                mask_bbox_rot[pixel_FN_rot] = mask_bbox_rot[pixel_FN_rot] + self.FN_epsilon
                mask_bbox_passat[pixel_FN_passat] = mask_bbox_passat[pixel_FN_passat] + self.FN_epsilon
                mask_bbox_rot_passat[pixel_FN_rot_passat] = mask_bbox_rot_passat[pixel_FN_rot_passat] + self.FN_epsilon
                mask_bbox_suv[pixel_FN_suv] = mask_bbox_suv[pixel_FN_suv] + self.FN_epsilon
                mask_bbox_rot_suv[pixel_FN_rot_suv] = mask_bbox_rot_suv[pixel_FN_rot_suv] + self.FN_epsilon
                mask_bbox_mpv[pixel_FN_mpv] = mask_bbox_mpv[pixel_FN_mpv] + self.FN_epsilon
                mask_bbox_rot_mpv[pixel_FN_rot_mpv] = mask_bbox_rot_mpv[pixel_FN_rot_mpv] + self.FN_epsilon

                tmp_loss = loss_function(mask_bbox, mask_target)
                tmp_loss_rot = loss_function(mask_bbox_rot, mask_target)
                tmp_loss_passat = loss_function(mask_bbox_passat, mask_target)
                tmp_loss_rot_passat = loss_function(mask_bbox_rot_passat, mask_target)
                tmp_loss_suv = loss_function(mask_bbox_suv, mask_target)
                tmp_loss_rot_suv = loss_function(mask_bbox_rot_suv, mask_target)
                tmp_loss_mpv = loss_function(mask_bbox_mpv, mask_target)
                tmp_loss_rot_mpv = loss_function(mask_bbox_rot_mpv, mask_target)

                tmp_loss = tmp_loss / torch.clip(torch.sum(mask_target), min=1)
                tmp_loss_rot = tmp_loss_rot / torch.clip(torch.sum(mask_target), min=1)
                tmp_loss_passat = tmp_loss_passat / torch.clip(torch.sum(mask_target), min=1)
                tmp_loss_rot_passat = tmp_loss_rot_passat / torch.clip(torch.sum(mask_target), min=1)
                tmp_loss_suv = tmp_loss_suv / torch.clip(torch.sum(mask_target), min=1)
                tmp_loss_rot_suv = tmp_loss_rot_suv / torch.clip(torch.sum(mask_target), min=1)
                tmp_loss_mpv = tmp_loss_mpv / torch.clip(torch.sum(mask_target), min=1)
                tmp_loss_rot_mpv = tmp_loss_rot_mpv / torch.clip(torch.sum(mask_target), min=1)

            all_losses = torch.tensor([tmp_loss, tmp_loss_rot, tmp_loss_passat, tmp_loss_rot_passat, tmp_loss_suv, tmp_loss_rot_suv, tmp_loss_mpv, tmp_loss_rot_mpv])
            lowest_idx = torch.argmin(all_losses)
            if lowest_idx == 0: #Then it is tmp_loss
                loss = loss + tmp_loss
            elif lowest_idx == 1: #Then it is tmp_loss_rot
                loss = loss + tmp_loss_rot
            elif lowest_idx == 2: #Then it is tmp_loss_passat
                loss = loss + tmp_loss_passat
            elif lowest_idx == 3: #Then it is tmp_loss_rot_passat
                loss = loss + tmp_loss_rot_passat
            elif lowest_idx == 4: #Then it is tmp_loss_suv
                loss = loss + tmp_loss_suv
            elif lowest_idx == 5: #Then it is tmp_loss_rot_suv
                loss = loss + tmp_loss_rot_suv
            elif lowest_idx == 6: #Then it is tmp_loss_mpv
                loss = loss + tmp_loss_mpv
            elif lowest_idx == 7: #Then it is tmp_loss_rot_mpv
                loss = loss + tmp_loss_rot_mpv

        loss = loss / bboxes.shape[0]

        return loss
    # End TCC-Det

    def get_box_cls_layer_loss(self, forward_ret_dict):
        loss_cfgs = self.model_cfg.LOSS_CONFIG
        rcnn_cls = forward_ret_dict['rcnn_cls']
        rcnn_cls_labels = forward_ret_dict['rcnn_cls_labels'].view(-1)
        if loss_cfgs.CLS_LOSS == 'BinaryCrossEntropy':
            rcnn_cls_flat = rcnn_cls.view(-1)
            batch_loss_cls = F.binary_cross_entropy(torch.sigmoid(rcnn_cls_flat), rcnn_cls_labels.float(), reduction='none')
            cls_valid_mask = (rcnn_cls_labels >= 0).float()
            rcnn_loss_cls = (batch_loss_cls * cls_valid_mask).sum() / torch.clamp(cls_valid_mask.sum(), min=1.0)
        elif loss_cfgs.CLS_LOSS == 'CrossEntropy':
            batch_loss_cls = F.cross_entropy(rcnn_cls, rcnn_cls_labels, reduction='none', ignore_index=-1)
            cls_valid_mask = (rcnn_cls_labels >= 0).float()
            rcnn_loss_cls = (batch_loss_cls * cls_valid_mask).sum() / torch.clamp(cls_valid_mask.sum(), min=1.0)
        else:
            raise NotImplementedError

        rcnn_loss_cls = rcnn_loss_cls * loss_cfgs.LOSS_WEIGHTS['rcnn_cls_weight']
        tb_dict = {'rcnn_loss_cls': rcnn_loss_cls.item()}
        return rcnn_loss_cls, tb_dict

    def get_loss(self, tb_dict=None):
        tb_dict = {} if tb_dict is None else tb_dict
        rcnn_loss = 0
        rcnn_loss_cls, cls_tb_dict = self.get_box_cls_layer_loss(self.forward_ret_dict)
        rcnn_loss += rcnn_loss_cls
        tb_dict.update(cls_tb_dict)

        # Start TCC-Det
        if self.use_template_loss:
            rcnn_loss_reg, reg_tb_dict = self.get_box_reg_layer_loss_custom(self.forward_ret_dict)
            rcnn_loss += rcnn_loss_reg
        if self.use_orig_loss:
            rcnn_loss_reg, reg_tb_dict = self.get_box_reg_layer_loss(self.forward_ret_dict)
            rcnn_loss += rcnn_loss_reg
        print(rcnn_loss)
        # End TCC-Det
        tb_dict.update(reg_tb_dict)
        tb_dict['rcnn_loss'] = rcnn_loss.item()

        return rcnn_loss, tb_dict

    def generate_predicted_boxes(self, batch_size, rois, cls_preds, box_preds):
        """
        Args:
            batch_size:
            rois: (B, N, 7)
            cls_preds: (BN, num_class)
            box_preds: (BN, code_size)

        Returns:

        """
        code_size = self.box_coder.code_size
        # batch_cls_preds: (B, N, num_class or 1)
        batch_cls_preds = cls_preds.view(batch_size, -1, cls_preds.shape[-1])
        batch_box_preds = box_preds.view(batch_size, -1, code_size)

        roi_ry = rois[:, :, 6].view(-1)
        roi_xyz = rois[:, :, 0:3].view(-1, 3)
        local_rois = rois.clone().detach()
        local_rois[:, :, 0:3] = 0

        batch_box_preds = self.box_coder.decode_torch(batch_box_preds, local_rois).view(-1, code_size)

        batch_box_preds = common_utils.rotate_points_along_z(
            batch_box_preds.unsqueeze(dim=1), roi_ry
        ).squeeze(dim=1)
        batch_box_preds[:, 0:3] += roi_xyz
        batch_box_preds = batch_box_preds.view(batch_size, -1, code_size)
        return batch_cls_preds, batch_box_preds
    # Start TCC-Det
    def print_grad(self, grad):
        grad = grad.detach().clone().cpu()

        if self.visu_lidar:
            self.custom_loader.vizualize_lidar(self.batch_dict)
            #self.custom_loader.vizualize_gt_bboxes(self.batch_dict)
            #self.custom_loader.vizualize_orig_gt_bboxes(self.batch_dict)

            max_bbox = torch.argmax(self.visu_scores, dim=0)
            print("HIGHEST SCORE: ", max_bbox)
            self.visu_batch_box = self.visu_batch_box[max_bbox, :]
            self.templates = self.templates[max_bbox]
            self.direction_templates = self.direction_templates[max_bbox]

            point_clouds = self.custom_loader.get_lidar_to_fit(int(self.batch_dict['frame_id'][0]), self.batch_dict, 0)
            point_clouds = point_clouds[self.correspondence[max_bbox]]

            chosen_template = self.templates[0, self.direction_templates.item(), :, :]

            distances = torch.cdist(point_clouds, chosen_template)

            closest_dist_temp_to_scan1, _ = torch.min(distances, dim=0)
            closest_dist_scan_to_temp1, _ = torch.min(distances, dim=1)

            closest_dist_temp_to_scan1 = 2*(torch.sigmoid(5. * closest_dist_temp_to_scan1) - 0.5)
            closest_dist_scan_to_temp1 = 2*(torch.sigmoid(5. * closest_dist_scan_to_temp1) - 0.5)

            self.color_temp = torch.zeros(closest_dist_scan_to_temp1.shape[0], 3)
            self.color_temp[:, 0] = closest_dist_scan_to_temp1
            self.color_temp[:, 1] = 1-closest_dist_scan_to_temp1


            self.custom_loader.vizualize_det_bboxes(self.visu_batch_box)
            if self.visu_type == 'step':
                self.custom_loader.vizualize_lidar_to_fit(point_clouds.unsqueeze(0), self.color_temp)
                self.custom_loader.vizualize_templates(self.templates, self.direction_templates)

            for i in range(self.visu_batch_box.shape[0]):
                print("Prediction: ", self.visu_batch_box[i].numpy()," Anchor: ", self.rois[i].detach().clone().cpu().numpy(), " Gradient: ", grad[i].numpy())
                #self.custom_loader.vizualize_gradient(self.visu_batch_box[i].numpy(), grad[i].numpy())
            self.custom_loader.vizualize_show(self.visu_type)
        else:
            for i in range(self.visu_batch_box.shape[0]):
                print("Prediction: ", self.visu_batch_box[i].numpy(), " Anchor: ",
                      self.anchors_pos[i].detach().clone().cpu().numpy(), " Gradient: ", grad[i].numpy())

    def print_grad_mask(self, grad):
        grad = grad.detach().clone().cpu()

        pixel_hit = torch.logical_and(self.mask > 0., self.mask_target > 0.5)
        pixel_car_nomask = torch.logical_and(self.mask > 0., self.mask_target < 0.5)
        pixel_nocar_mask = torch.logical_and(self.mask == 0., self.mask_target > 0.5)
        pixel_truenegative = torch.logical_and(self.mask == 0., self.mask_target < 0.5)

        print("Gradient TP")
        print(grad[pixel_hit])
        print("Values TP")
        print(self.mask[pixel_hit])
        print(self.mask_target[pixel_hit])
        print("Gradient FP")
        print(grad[pixel_car_nomask])
        print("Values FP")
        print(self.mask[pixel_car_nomask])
        print(self.mask_target[pixel_car_nomask])
        print("Gradient FN")
        print(grad[pixel_nocar_mask])
        print("Values FN")
        print(self.mask[pixel_nocar_mask])
        print(self.mask_target[pixel_nocar_mask])
        print("Gradient TN")
        print(grad[pixel_truenegative])
        print("Values TN")
        print(self.mask[pixel_truenegative])
        print(self.mask_target[pixel_truenegative])

    def print_grad_mask_distloss(self, grad):
        grad = grad.detach().clone().cpu()

        pixel_hit = torch.logical_and(self.mask < 99., self.mask_target < 99.)
        pixel_car_nomask = torch.logical_and(self.mask < 99., self.mask_target > 99.)
        pixel_nocar_mask = torch.logical_and(self.mask > 99., self.mask_target < 99.)
        pixel_truenegative = torch.logical_and(self.mask > 99., self.mask_target > 99.)

        print("Gradient TP")
        print(grad[0, ..., 0][pixel_hit])
        print("Values TP")
        print(self.mask[pixel_hit])
        print(self.mask_target[pixel_hit])
        print("Gradient FP")
        print(grad[0, ..., 0][pixel_car_nomask])
        print("Values FP")
        print(self.mask[pixel_car_nomask])
        print(self.mask_target[pixel_car_nomask])
        print("Gradient FN")
        print(grad[0, ..., 0][pixel_nocar_mask])
        print("Values FN")
        print(self.mask[pixel_nocar_mask])
        print(self.mask_target[pixel_nocar_mask])
        print("Gradient TN")
        print(grad[0, ..., 0][pixel_truenegative])
        print("Values TN")
        print(self.mask[pixel_truenegative])
        print(self.mask_target[pixel_truenegative])

    def print_grad_mask_plot(self, grad):
        plt.figure(figsize=(10, 10))
        x = (torch.abs(grad) > 0.)
        plt.imshow(x.type(torch.uint8).cpu().detach().numpy())
        plt.axis("off")
        plt.show()
    # End TCC-Det
