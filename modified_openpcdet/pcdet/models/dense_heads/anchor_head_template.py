# Modified by: Jan Skvrna for the purpose of the TCC-Det
# Modified parts are marked with the comment: # Start TCC-Det and # End TCC-Det

import numpy as np
import torch
import torch.nn as nn
# Start TCC-Det
from scipy.spatial.distance import cdist
import time
# End TCC-Det

from ...utils import box_coder_utils, common_utils, loss_utils
from .target_assigner.anchor_generator import AnchorGenerator
from .target_assigner.atss_target_assigner import ATSSTargetAssigner
from .target_assigner.axis_aligned_target_assigner import AxisAlignedTargetAssigner

class AnchorHeadTemplate(nn.Module):
    def __init__(self, model_cfg, num_class, class_names, grid_size, point_cloud_range, predict_boxes_when_training):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_class = num_class
        self.class_names = class_names
        self.predict_boxes_when_training = predict_boxes_when_training
        self.use_multihead = self.model_cfg.get('USE_MULTIHEAD', False)
        # Start TCC-Det
        self.use_template_loss = self.model_cfg.get('USE_TEMPLATE_LOSS', False)
        self.disable_extent = self.model_cfg.get('DISABLE_EXTENT', False)
        self.filter_predictions = self.model_cfg.get('FILTER_PREDICTION_METHOD', None)
        self.data_dict = None
        self.visu_batch_box = None
        self.point_clouds_to_fit = None
        self.templates = None
        self.direction_templates = None
        self.anchors_pos = None
        self.counter = 0
        self.mask_loss_div = 2. # TODO Should come from the config file

        self.print_gradients = False
        self.visu_lidar = False
        self.visu_masks = False
        self.visu_type = "step" #Possible: animation, step
        self.grad_visu_type = "all" #Possible: all, step

        self.use_BCELoss = False
        self.use_L1Loss = self.model_cfg.get('LOSS_L1', False)
        self.use_L2Loss = self.model_cfg.get('LOSS_L2', False)
        self.use_SmoothL1Loss = self.model_cfg.get('LOSS_SMOOTHL1', False)

        np.set_printoptions(formatter={'float': lambda x: "{0:0.4f}".format(x)})

        self.smoothL1 = nn.SmoothL1Loss()
        # End TCC-Det

        anchor_target_cfg = self.model_cfg.TARGET_ASSIGNER_CONFIG
        self.box_coder = getattr(box_coder_utils, anchor_target_cfg.BOX_CODER)(
            num_dir_bins=anchor_target_cfg.get('NUM_DIR_BINS', 6),
            **anchor_target_cfg.get('BOX_CODER_CONFIG', {})
        )

        anchor_generator_cfg = self.model_cfg.ANCHOR_GENERATOR_CONFIG
        anchors, self.num_anchors_per_location = self.generate_anchors(
            anchor_generator_cfg, grid_size=grid_size, point_cloud_range=point_cloud_range,
            anchor_ndim=self.box_coder.code_size
        )
        self.anchors = [x.cuda() for x in anchors]
        self.target_assigner = self.get_target_assigner(anchor_target_cfg)

        self.forward_ret_dict = {}
        self.build_losses(self.model_cfg.LOSS_CONFIG)


    @staticmethod
    def generate_anchors(anchor_generator_cfg, grid_size, point_cloud_range, anchor_ndim=7):
        anchor_generator = AnchorGenerator(
            anchor_range=point_cloud_range,
            anchor_generator_config=anchor_generator_cfg
        )
        feature_map_size = [grid_size[:2] // config['feature_map_stride'] for config in anchor_generator_cfg]
        anchors_list, num_anchors_per_location_list = anchor_generator.generate_anchors(feature_map_size)

        if anchor_ndim != 7:
            for idx, anchors in enumerate(anchors_list):
                pad_zeros = anchors.new_zeros([*anchors.shape[0:-1], anchor_ndim - 7])
                new_anchors = torch.cat((anchors, pad_zeros), dim=-1)
                anchors_list[idx] = new_anchors

        return anchors_list, num_anchors_per_location_list

    def get_target_assigner(self, anchor_target_cfg):
        if anchor_target_cfg.NAME == 'ATSS':
            target_assigner = ATSSTargetAssigner(
                topk=anchor_target_cfg.TOPK,
                box_coder=self.box_coder,
                use_multihead=self.use_multihead,
                match_height=anchor_target_cfg.MATCH_HEIGHT
            )
        elif anchor_target_cfg.NAME == 'AxisAlignedTargetAssigner':
            target_assigner = AxisAlignedTargetAssigner(
                model_cfg=self.model_cfg,
                class_names=self.class_names,
                box_coder=self.box_coder,
                match_height=anchor_target_cfg.MATCH_HEIGHT
            )
        else:
            raise NotImplementedError
        return target_assigner

    def build_losses(self, losses_cfg):
        self.add_module(
            'cls_loss_func',
            loss_utils.SigmoidFocalClassificationLoss(alpha=0.25, gamma=2.0)
        )
        reg_loss_name = 'WeightedSmoothL1Loss' if losses_cfg.get('REG_LOSS_TYPE', None) is None \
            else losses_cfg.REG_LOSS_TYPE
        self.add_module(
            'reg_loss_func',
            getattr(loss_utils, reg_loss_name)(code_weights=losses_cfg.LOSS_WEIGHTS['code_weights'])
        )
        self.add_module(
            'dir_loss_func',
            loss_utils.WeightedCrossEntropyLoss()
        )

    def assign_targets(self, gt_boxes):
        """
        Args:
            gt_boxes: (B, M, 8)
        Returns:

        """
        targets_dict = self.target_assigner.assign_targets(
            self.anchors, gt_boxes
        )
        return targets_dict

    def get_cls_layer_loss(self):
        cls_preds = self.forward_ret_dict['cls_preds']
        box_cls_labels = self.forward_ret_dict['box_cls_labels']
        batch_size = int(cls_preds.shape[0])
        cared = box_cls_labels >= 0  # [N, num_anchors]
        positives = box_cls_labels > 0
        negatives = box_cls_labels == 0
        negative_cls_weights = negatives * 1.0
        cls_weights = (negative_cls_weights + 1.0 * positives).float()
        reg_weights = positives.float()
        if self.num_class == 1:
            # class agnostic
            box_cls_labels[positives] = 1

        pos_normalizer = positives.sum(1, keepdim=True).float()
        reg_weights /= torch.clamp(pos_normalizer, min=1.0)
        cls_weights /= torch.clamp(pos_normalizer, min=1.0)
        cls_targets = box_cls_labels * cared.type_as(box_cls_labels)
        cls_targets = cls_targets.unsqueeze(dim=-1)

        cls_targets = cls_targets.squeeze(dim=-1)
        one_hot_targets = torch.zeros(
            *list(cls_targets.shape), self.num_class + 1, dtype=cls_preds.dtype, device=cls_targets.device
        )
        one_hot_targets.scatter_(-1, cls_targets.unsqueeze(dim=-1).long(), 1.0)
        cls_preds = cls_preds.view(batch_size, -1, self.num_class)
        one_hot_targets = one_hot_targets[..., 1:]
        cls_loss_src = self.cls_loss_func(cls_preds, one_hot_targets, weights=cls_weights)  # [N, M]
        cls_loss = cls_loss_src.sum() / batch_size

        cls_loss = cls_loss * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['cls_weight']
        tb_dict = {
            'rpn_loss_cls': cls_loss.item()
        }
        return cls_loss, tb_dict

    @staticmethod
    def add_sin_difference(boxes1, boxes2, dim=6):
        assert dim != -1
        rad_pred_encoding = torch.sin(boxes1[..., dim:dim + 1]) * torch.cos(boxes2[..., dim:dim + 1])
        rad_tg_encoding = torch.cos(boxes1[..., dim:dim + 1]) * torch.sin(boxes2[..., dim:dim + 1])
        boxes1 = torch.cat([boxes1[..., :dim], rad_pred_encoding, boxes1[..., dim + 1:]], dim=-1)
        boxes2 = torch.cat([boxes2[..., :dim], rad_tg_encoding, boxes2[..., dim + 1:]], dim=-1)
        return boxes1, boxes2

    @staticmethod
    def get_direction_target(anchors, reg_targets, one_hot=True, dir_offset=0, num_bins=2):
        batch_size = reg_targets.shape[0]
        anchors = anchors.view(batch_size, -1, anchors.shape[-1])
        rot_gt = reg_targets[..., 6] + anchors[..., 6]
        offset_rot = common_utils.limit_period(rot_gt - dir_offset, 0, 2 * np.pi)
        dir_cls_targets = torch.floor(offset_rot / (2 * np.pi / num_bins)).long()
        dir_cls_targets = torch.clamp(dir_cls_targets, min=0, max=num_bins - 1)

        if one_hot:
            dir_targets = torch.zeros(*list(dir_cls_targets.shape), num_bins, dtype=anchors.dtype,
                                      device=dir_cls_targets.device)
            dir_targets.scatter_(-1, dir_cls_targets.unsqueeze(dim=-1).long(), 1.0)
            dir_cls_targets = dir_targets
        return dir_cls_targets

    def get_box_reg_layer_loss(self):
        box_preds = self.forward_ret_dict['box_preds']
        box_dir_cls_preds = self.forward_ret_dict.get('dir_cls_preds', None)
        box_reg_targets = self.forward_ret_dict['box_reg_targets']
        box_cls_labels = self.forward_ret_dict['box_cls_labels']
        batch_size = int(box_preds.shape[0])

        positives = box_cls_labels > 0
        reg_weights = positives.float()
        pos_normalizer = positives.sum(1, keepdim=True).float()
        reg_weights /= torch.clamp(pos_normalizer, min=1.0)

        if isinstance(self.anchors, list):
            if self.use_multihead:
                anchors = torch.cat(
                    [anchor.permute(3, 4, 0, 1, 2, 5).contiguous().view(-1, anchor.shape[-1]) for anchor in
                     self.anchors], dim=0)
            else:
                anchors = torch.cat(self.anchors, dim=-3)
        else:
            anchors = self.anchors
        anchors = anchors.view(1, -1, anchors.shape[-1]).repeat(batch_size, 1, 1)
        box_preds = box_preds.view(batch_size, -1,
                                   box_preds.shape[-1] // self.num_anchors_per_location if not self.use_multihead else
                                   box_preds.shape[-1])

        # sin(a - b) = sinacosb-cosasinb
        box_preds_sin, reg_targets_sin = self.add_sin_difference(box_preds, box_reg_targets)
        loc_loss_src = self.reg_loss_func(box_preds_sin, reg_targets_sin, weights=reg_weights)  # [N, M]
        loc_loss = loc_loss_src.sum() / batch_size

        loc_loss = loc_loss * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['loc_weight']
        box_loss = loc_loss
        tb_dict = {
            'rpn_loss_loc': loc_loss.item()
        }

        if box_dir_cls_preds is not None:
            dir_targets = self.get_direction_target(
                anchors, box_reg_targets,
                dir_offset=self.model_cfg.DIR_OFFSET,
                num_bins=self.model_cfg.NUM_DIR_BINS
            )

            dir_logits = box_dir_cls_preds.view(batch_size, -1, self.model_cfg.NUM_DIR_BINS)
            weights = positives.type_as(dir_logits)
            weights /= torch.clamp(weights.sum(-1, keepdim=True), min=1.0)
            dir_loss = self.dir_loss_func(dir_logits, dir_targets, weights=weights)
            dir_loss = dir_loss.sum() / batch_size
            dir_loss = dir_loss * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['dir_weight']
            box_loss += dir_loss
            tb_dict['rpn_loss_dir'] = dir_loss.item()

        return box_loss, tb_dict

    # Start TCC-Det
    def get_custom_box_reg_layer_loss(self):
        box_cls_labels = self.forward_ret_dict['box_cls_labels']
        batch_size = int(box_cls_labels.shape[0])
        batch_box_preds = self.data_dict['batch_box_preds_anchors']

        correspondence = None

        loss = torch.tensor([0.]).cuda()

        for i in range(batch_size):
            positives = box_cls_labels[i] > 0
            # Create an index tensor for the positives using the mask
            positives_idx = torch.nonzero(positives)[:, 0]

            batch_box_preds_pos = batch_box_preds[i, positives_idx[:], :]
            anchors_batch = self.anchors[0].view(-1, self.anchors[0].shape[-1])
            anchors_pos = anchors_batch[positives_idx[:], :]

            reg_weights = torch.ones(batch_box_preds_pos.shape[0]).cuda()
            pos_normalizer = positives.sum(0, keepdim=True).float()
            reg_weights /= torch.clamp(pos_normalizer, min=1.0)

            if self.print_gradients:
                batch_box_preds_pos.register_hook(self.print_grad)

            # Lets get the point-clouds for this frame
            point_clouds = self.custom_loader.get_lidar_to_fit(int(self.data_dict['frame_id'][i]), self.data_dict, i)

            # Compute all point-cloud centers for the matching with the detections
            pcloud_centers = torch.zeros((len(point_clouds), 3), device='cuda')
            for z in range(len(point_clouds)):
                pcloud_centers[z, :] = torch.median(point_clouds[z], dim=0)[0]

            # Lets find closest point-cloud for each bbox
            if batch_box_preds_pos.shape[0] == 0 or len(point_clouds) == 0:
                continue
            distances = torch.cdist(batch_box_preds_pos[:, 0:3], pcloud_centers)
            correspondence = torch.argmin(distances, dim=1)

            # Recompute the moving cars angles, becaue of the data augmentation
            self.custom_loader.recompute_moving_cars_angles(self.data_dict, int(self.data_dict['frame_id'][i]), i)
            # Get masks which says if the detections are moving or standing
            standing_moving_info = self.custom_loader.get_standing_moving_info(self.data_dict, correspondence, int(self.data_dict['frame_id'][i]))
            # Get the angles for the moving cars
            moving_cars_angles = self.custom_loader.get_moving_cars_angles(self.data_dict, correspondence, int(self.data_dict['frame_id'][i]))

            # Now lets generate template for each bbox prediction
            templates = self.custom_loader.get_templates_from_anchors(batch_box_preds_pos, anchors_pos, standing_moving_info, moving_cars_angles)

            tmp_loss = torch.tensor([0.]).cuda()

            template_loss, direction_templates = self.compute_template_loss(templates, point_clouds, correspondence,
                                                             standing_moving_info, moving_cars_angles,
                                                             batch_box_preds_pos)

            mask_loss = self.compute_mask_loss_all(int(self.data_dict['frame_id'][i]), batch_box_preds_pos, correspondence, i)

            tmp_loss = tmp_loss + template_loss
            tmp_loss = tmp_loss + mask_loss / self.mask_loss_div
            loss += tmp_loss

            if self.visu_lidar and not self.print_gradients:
                self.custom_loader.vizualize_lidar(self.data_dict)
                self.custom_loader.vizualize_gt_bboxes(self.data_dict)
                self.custom_loader.vizualize_orig_gt_bboxes(self.data_dict)
                self.custom_loader.vizualize_det_bboxes(batch_box_preds_pos[:, :])
                if self.visu_type == 'step':
                    self.custom_loader.vizualize_lidar_to_fit(point_clouds)
                    self.custom_loader.vizualize_templates(templates, direction_templates)
                self.custom_loader.vizualize_show(self.visu_type)
            elif self.visu_lidar or self.print_gradients:
                self.visu_batch_box = batch_box_preds_pos.detach().clone().cpu()
                self.point_clouds_to_fit = point_clouds
                self.templates = templates
                self.direction_templates = direction_templates
                self.anchors_pos = anchors_pos

        loss = loss / batch_size
        tb_dict = {
            'rpn_loss_loc': loss.item()
        }

        box_dir_cls_preds = self.forward_ret_dict.get('dir_cls_preds', None)
        if box_dir_cls_preds is not None:
            box_reg_targets = self.forward_ret_dict['box_reg_targets']
            positives = box_cls_labels > 0
            if isinstance(self.anchors, list):
                if self.use_multihead:
                    anchors = torch.cat(
                        [anchor.permute(3, 4, 0, 1, 2, 5).contiguous().view(-1, anchor.shape[-1]) for anchor in
                         self.anchors], dim=0)
                else:
                    anchors = torch.cat(self.anchors, dim=-3)
            else:
                anchors = self.anchors
            anchors = anchors.view(1, -1, anchors.shape[-1]).repeat(batch_size, 1, 1)
            dir_targets = self.get_direction_target(
                anchors, box_reg_targets,
                dir_offset=self.model_cfg.DIR_OFFSET,
                num_bins=self.model_cfg.NUM_DIR_BINS
            )

            dir_logits = box_dir_cls_preds.view(batch_size, -1, self.model_cfg.NUM_DIR_BINS)
            weights = positives.type_as(dir_logits)
            weights /= torch.clamp(weights.sum(-1, keepdim=True), min=1.0)
            dir_loss = self.dir_loss_func(dir_logits, dir_targets, weights=weights)
            dir_loss = dir_loss.sum() / batch_size
            dir_loss = dir_loss * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['dir_weight']
            loss += dir_loss
            tb_dict['rpn_loss_dir'] = dir_loss.item()

        print(self.data_dict['frame_id'], loss)
        return loss, tb_dict

    def compute_template_loss(self, templates, point_clouds, correspondence, standing_moving_info, moving_cars_angles, batch_box_preds_pos):
        tmp_loss = torch.tensor([0.]).cuda()
        direction_templates = torch.zeros(batch_box_preds_pos.shape[0], dtype=torch.int8)
        for z in range(templates.shape[0]):
            tmp_loss_front = torch.tensor([0.]).cuda()
            tmp_loss_back = torch.tensor([0.]).cuda()

            distances1 = torch.cdist(point_clouds[correspondence[z]], templates[z, 0])
            distances2 = torch.cdist(point_clouds[correspondence[z]], templates[z, 1])

            closest_dist_temp_to_scan1, _ = torch.min(distances1, dim=0)
            closest_dist_scan_to_temp1, _ = torch.min(distances1, dim=1)

            closest_dist_temp_to_scan2, _ = torch.min(distances2, dim=0)
            closest_dist_scan_to_temp2, _ = torch.min(distances2, dim=1)

            closest_dist_temp_to_scan1 = torch.sigmoid(5. * closest_dist_temp_to_scan1) - 0.5
            closest_dist_scan_to_temp1 = torch.sigmoid(5. * closest_dist_scan_to_temp1) - 0.5
            closest_dist_temp_to_scan2 = torch.sigmoid(5. * closest_dist_temp_to_scan2) - 0.5
            closest_dist_scan_to_temp2 = torch.sigmoid(5. * closest_dist_scan_to_temp2) - 0.5

            tmp_loss_front += torch.sum(closest_dist_temp_to_scan1) / templates[z, 0].shape[0]
            tmp_loss_front += torch.sum(closest_dist_scan_to_temp1) / point_clouds[correspondence[z]].shape[0]
            tmp_loss_back += torch.sum(closest_dist_temp_to_scan2) / templates[z, 1].shape[0]
            tmp_loss_back += torch.sum(closest_dist_scan_to_temp2) / point_clouds[correspondence[z]].shape[0]

            tmp_loss_front = tmp_loss_front
            tmp_loss_back = tmp_loss_back

            if standing_moving_info[z] == 1 and moving_cars_angles[z] is not None:
                rad_pred_encoding = torch.sin(batch_box_preds_pos[z, 6]) * torch.cos(
                    torch.tensor(moving_cars_angles[z]).cuda())
                rad_tg_encoding = torch.cos(batch_box_preds_pos[z, 6]) * torch.sin(
                    torch.tensor(moving_cars_angles[z]).cuda())

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

        self.custom_loader.init_rendering(frame_idx)

        loss = torch.tensor([0.]).cuda()

        for i in range(bboxes.shape[0]):
            if self.use_BCELoss:
                mask_target, mask_target_weights = self.custom_loader.get_target_car_mask(frame_idx, correspondence[i])

                mask_bbox, mask_bbox_visu = self.custom_loader.render_bbox_to_mask_single(bboxes[i], self.data_dict, frame_idx, batch_idx, self.visu_masks)
                mask_bbox_rot, mask_bbox_visu_rot = self.custom_loader.render_bbox_to_mask_single(bboxes_rot[i], self.data_dict, frame_idx, batch_idx, self.visu_masks)

                mask_loss = torch.nn.BCELoss(mask_target_weights)

                tmp_loss = mask_loss(mask_bbox[0, :, :, 0], mask_target)
                tmp_loss_rot = mask_loss(mask_bbox_rot[0, :, :, 0], mask_target)
            elif self.use_L1Loss or self.use_L2Loss or self.use_SmoothL1Loss:
                mask_target, mask_target_weights, mask_target_visu = self.custom_loader.get_target_car_mask_distloss(frame_idx, correspondence[i], bboxes[i])

                #mask_bbox, mask_bbox_visu = self.custom_loader.rasterize_bbox_to_mask_single(bboxes[i], self.data_dict, frame_idx, batch_idx, self.visu_masks)
                mask_bbox, mask_bbox_visu = self.custom_loader.render_bbox_to_mask_single_distloss(bboxes[i], self.data_dict, frame_idx, batch_idx, self.visu_masks)
                mask_bbox_rot, mask_bbox_visu_rot = self.custom_loader.render_bbox_to_mask_single_distloss(bboxes_rot[i], self.data_dict, frame_idx, batch_idx, self.visu_masks)

                diff_mask_bbox = mask_bbox[0, :, :, 0] - mask_target
                diff_mask_bbox_rot = mask_bbox_rot[0, :, :, 0] - mask_target

                diff_mask_bbox = torch.mul(diff_mask_bbox, mask_target_weights)
                diff_mask_bbox_rot = torch.mul(diff_mask_bbox_rot, mask_target_weights)

                if self.use_L1Loss:
                    loss_function = nn.L1Loss(reduction='sum')
                elif self.use_L2Loss:
                    loss_function = nn.MSELoss(reduction='sum')
                elif self.use_SmoothL1Loss:
                    loss_function = nn.SmoothL1Loss(reduction='sum')


                tmp_loss = loss_function(diff_mask_bbox, torch.zeros_like(diff_mask_bbox))
                tmp_loss_rot = loss_function(diff_mask_bbox_rot, torch.zeros_like(diff_mask_bbox))

                tmp_loss = tmp_loss / torch.clip(torch.sum(mask_target < 99.), min=1)
                tmp_loss_rot = tmp_loss_rot / torch.clip(torch.sum(mask_target < 99.), min=1)

                tmp_loss = tmp_loss / torch.clip(loss_function(100. - bboxes[i, 0], torch.tensor(0.).cuda()), min=1)
                tmp_loss_rot = tmp_loss_rot / torch.clip(loss_function(100. - bboxes[i, 0], torch.tensor(0.).cuda()), min=1)

            if tmp_loss < tmp_loss_rot:
                loss = loss + tmp_loss
                #print(tmp_loss)

                if self.visu_masks:
                    self.custom_loader.vizualize_image_masks(frame_idx, [mask_target_visu], [mask_bbox_visu])
            else:
                loss = loss + tmp_loss_rot
                #print(tmp_loss_rot)

                if self.visu_masks:
                    self.custom_loader.vizualize_image_masks(frame_idx, [mask_target_visu], [mask_bbox_visu_rot])

        return loss
    # End TCC-Det
    def get_loss(self):
        cls_loss, tb_dict = self.get_cls_layer_loss()
        # Start TCC-Det
        if self.use_template_loss:
            box_loss, tb_dict_box = self.get_custom_box_reg_layer_loss()
        else:
            box_loss, tb_dict_box = self.get_box_reg_layer_loss()
        # End TCC-Det
        tb_dict.update(tb_dict_box)
        rpn_loss = cls_loss + box_loss

        tb_dict['rpn_loss'] = rpn_loss.item()
        return rpn_loss, tb_dict

    def generate_predicted_boxes(self, batch_size, cls_preds, box_preds, dir_cls_preds=None):
        """
        Args:
            batch_size:
            cls_preds: (N, H, W, C1)
            box_preds: (N, H, W, C2)
            dir_cls_preds: (N, H, W, C3)

        Returns:
            batch_cls_preds: (B, num_boxes, num_classes)
            batch_box_preds: (B, num_boxes, 7+C)

        """
        if isinstance(self.anchors, list):
            if self.use_multihead:
                anchors = torch.cat([anchor.permute(3, 4, 0, 1, 2, 5).contiguous().view(-1, anchor.shape[-1])
                                     for anchor in self.anchors], dim=0)
            else:
                anchors = torch.cat(self.anchors, dim=-3)
        else:
            anchors = self.anchors
        num_anchors = anchors.view(-1, anchors.shape[-1]).shape[0]
        batch_anchors = anchors.view(1, -1, anchors.shape[-1]).repeat(batch_size, 1, 1)
        batch_cls_preds = cls_preds.view(batch_size, num_anchors, -1).float() \
            if not isinstance(cls_preds, list) else cls_preds
        batch_box_preds = box_preds.view(batch_size, num_anchors, -1) if not isinstance(box_preds, list) \
            else torch.cat(box_preds, dim=1).view(batch_size, num_anchors, -1)
        batch_box_preds = self.box_coder.decode_torch(batch_box_preds, batch_anchors)

        if dir_cls_preds is not None:
            dir_offset = self.model_cfg.DIR_OFFSET
            dir_limit_offset = self.model_cfg.DIR_LIMIT_OFFSET
            dir_cls_preds = dir_cls_preds.view(batch_size, num_anchors, -1) if not isinstance(dir_cls_preds, list) \
                else torch.cat(dir_cls_preds, dim=1).view(batch_size, num_anchors, -1)
            dir_labels = torch.max(dir_cls_preds, dim=-1)[1]

            period = (2 * np.pi / self.model_cfg.NUM_DIR_BINS)
            dir_rot = common_utils.limit_period(
                batch_box_preds[..., 6] - dir_offset, dir_limit_offset, period
            )
            batch_box_preds[..., 6] = dir_rot + dir_offset + period * dir_labels.to(batch_box_preds.dtype)

        if isinstance(self.box_coder, box_coder_utils.PreviousResidualDecoder):
            batch_box_preds[..., 6] = common_utils.limit_period(
                -(batch_box_preds[..., 6] + np.pi / 2), offset=0.5, period=np.pi * 2
            )

        return batch_cls_preds, batch_box_preds

    def forward(self, **kwargs):
        raise NotImplementedError

    # Start TCC-Det
    def print_grad(self, grad):
        grad = grad.detach().clone().cpu()

        if self.visu_lidar and self.counter % 1 == 0:
            self.custom_loader.vizualize_lidar(self.data_dict)
            self.custom_loader.vizualize_gt_bboxes(self.data_dict)
            self.custom_loader.vizualize_orig_gt_bboxes(self.data_dict)
            self.custom_loader.vizualize_det_bboxes(self.visu_batch_box[:, :])
            if self.visu_type == 'step':
                self.custom_loader.vizualize_lidar_to_fit(self.point_clouds_to_fit)
                self.custom_loader.vizualize_templates(self.templates, self.direction_templates)

            for i in range(self.visu_batch_box.shape[0]):
                print("Prediction: ", self.visu_batch_box[i].numpy()," Anchor: ", self.anchors_pos[i].detach().clone().cpu().numpy(), " Gradient: ", grad[i].numpy())
                self.custom_loader.vizualize_gradient(self.visu_batch_box[i].numpy(), grad[i].numpy())
            self.custom_loader.vizualize_show(self.visu_type)
        else:
            for i in range(self.visu_batch_box.shape[0]):
                print("Prediction: ", self.visu_batch_box[i].numpy(), " Anchor: ",
                      self.anchors_pos[i].detach().clone().cpu().numpy(), " Gradient: ", grad[i].numpy())
        self.counter += 1
    # End TCC-Det
