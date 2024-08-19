import pyvista as pv
import numpy as np
import matplotlib.pyplot as plt
import random
from base_class import BaseClass
import tensorflow.compat.v1 as tf
tf.enable_eager_execution()

from waymo_open_dataset.utils import range_image_utils
from waymo_open_dataset.utils import transform_utils
from waymo_open_dataset.utils import frame_utils
from waymo_open_dataset import dataset_pb2 as open_dataset
class Vizualizer(BaseClass):
    def __init__(self, config_path):
        super().__init__(config_path)
        self.showed = False

    def vizu_show(self, visu_type):
        if visu_type == "animation":
            if self.showed == False:
                self.plotter.set_background('black')
                self.plotter.camera_position = 'xy'
                self.plotter.show(interactive_update=True)
            else:
                self.plotter.update()
            self.plotter.clear()
        elif visu_type == "step":
            self.plotter.set_background('white')
            self.plotter.camera_position = 'xy'
            self.plotter.show()
            self.plotter = pv.Plotter()
        else:
            print("UNKNOWN TYPE FOR VISU")

    def vizu_lidar(self, data_dict):
        #print(data_dict['frame_id'][0])
        #rint(data_dict['gt_boxes'][0])
        if self.plotter is None:
            self.plotter = pv.Plotter()
        cloud_mesh = pv.PolyData(data_dict['points'][:, 1:4].cpu().numpy())
        self.plotter.add_points(cloud_mesh, color='black', point_size=5)

    def vizu_gt_bbox(self, data_dict):
        gt_boxes = data_dict['gt_boxes'][0]

        for i in range(gt_boxes.shape[0]):
            bbox_location = gt_boxes[i][:3].cpu().numpy()
            bbox_extents = gt_boxes[i][3:6].cpu().numpy()
            bbox_yaw = gt_boxes[i][6].cpu().numpy()

            bbox = pv.Box(bounds=[-bbox_extents[0]/2., bbox_extents[0]/2., -bbox_extents[1]/2.,
                                  bbox_extents[1]/2., -bbox_extents[2]/2., bbox_extents[2]/2.]).outline()

            bbox = bbox.rotate_z(np.rad2deg(bbox_yaw))

            bbox = bbox.translate(bbox_location)

            self.plotter.add_mesh(bbox, color='blue', line_width=1)

    def vizu_orig_gt_bbox(self, data_dict):
        #print(data_dict['frame_id'])
        f = open(self.label_path + data_dict['frame_id'][0] + '.txt', 'r')
        lines = f.readlines()
        arr = [line.strip().split(" ") for line in lines]
        for i in range(len(arr)):
            if arr[i][0] == 'Car' or arr[i][0] == 'car' or arr[i][0] == 'Van' or arr[i][0] == 'van':
                #print(arr[i])
                size = np.array(
                    [(float(arr[i][9])), (float(arr[i][8])), (float(arr[i][10]))])  # Height, width, length
                center = np.array(
                    [(float(arr[i][11])), (float(arr[i][12]) - size[1] / 2), (float(arr[i][13]))]).reshape((1,3))  # x,y,z
                center = self.transform_lidar_cam2_to_velo(center).T
                if 'flip_x' in data_dict and data_dict['flip_x'] > 0.5:
                    center[1] = -1 * center[1]
                center = np.matmul(data_dict['lidar_aug_matrix'][0, :3, :3].cpu().numpy(), center)
                center = center[:3, 0] + data_dict['lidar_aug_matrix'][0, :3, 3].cpu().numpy()
                yaw = (float(arr[i][14])) - np.pi / 2. 

                bbox = pv.Box(bounds=[-size[2] / 2., size[2] / 2., -size[1] / 2.,
                                      size[1] / 2., -size[0] / 2., size[0] / 2.]).outline()

                if 'flip_x' in data_dict and 'noise_rot' in data_dict and data_dict['flip_x'] > 0.5:
                    bbox = bbox.rotate_z(np.rad2deg(+yaw + data_dict['noise_rot'].cpu().item()))
                elif 'noise_rot' in data_dict:
                    bbox = bbox.rotate_z(np.rad2deg(-yaw + data_dict['noise_rot'].cpu().item()))
                else:
                    bbox = bbox.rotate_z(np.rad2deg(-yaw))

                bbox = bbox.translate(center)

                self.plotter.add_mesh(bbox, color='green', line_width=1)

    def vizu_det_bbox(self, det_bbox):
        for i in range(det_bbox.shape[0]):
            bbox_location = det_bbox[i][:3].detach().cpu().numpy()
            bbox_extents = det_bbox[i][3:6].detach().cpu().numpy()
            bbox_yaw = det_bbox[i][6].detach().cpu().numpy()

            bbox = pv.Box(bounds=[-bbox_extents[0]/2., bbox_extents[0]/2., -bbox_extents[1]/2.,
                                  bbox_extents[1]/2., -bbox_extents[2]/2., bbox_extents[2]/2.]).outline()

            bbox = bbox.rotate_z(np.rad2deg(bbox_yaw))

            bbox = bbox.translate(bbox_location)

            self.plotter.add_mesh(bbox, color='red', line_width=1)

    def vizu_lidar_to_fit(self, lidar_to_fit, colors):
        for i in range(len(lidar_to_fit)):
            cloud_mesh = pv.PolyData(lidar_to_fit[i].cpu().numpy())
            if colors is None:
                self.plotter.add_points(cloud_mesh, point_size=10, color='blue')
            else:
                self.plotter.add_points(cloud_mesh, point_size=15, scalars=np.array(colors), rgb=True)


    def vizu_templates(self, templates, direction, colors):
        for i in range(templates.shape[0]):
            cloud_mesh = pv.PolyData(templates[i, direction[i], :, :].detach().cpu().numpy())
            if colors is None:
                self.plotter.add_points(cloud_mesh, point_size=10, color='blue')
            else:
                self.plotter.add_points(cloud_mesh, point_size=15, scalars=np.array(colors), rgb=True)

    def vizu_gradient(self, loc, grad):
        pos_grad_arrow_start = loc[:3]
        pos_grad_arrow_end = -grad[:3]

        arrow = pv.Arrow(pos_grad_arrow_start, pos_grad_arrow_end, shaft_radius=0.03, tip_length=0.25, tip_radius=0.05)
        self.plotter.add_mesh(arrow, color='red')

        loc[4] = -np.sign(grad[4]) * loc[4]
        rot_grad_arrow_start = loc[:3] + np.array([np.cos(loc[6]) * loc[3] - np.sin(loc[6]) * loc[4], np.sin(loc[6]) * loc[3] + np.cos(loc[6]) * loc[4], loc[5]]) / 2.
        rot_grad_arrow_start2 = loc[:3] - np.array([np.cos(loc[6]) * loc[3] - np.sin(loc[6]) * loc[4],  np.sin(loc[6]) * loc[3] + np.cos(loc[6]) * loc[4], -loc[5]]) / 2.
        angle = loc[6] - (np.pi/2.) * np.sign(grad[6])
        angle2 = loc[6] + (np.pi / 2.) * np.sign(grad[6])

        rot_grad_arrow_end = np.array([np.cos(angle), np.sin(angle), 0])
        rot_grad_arrow_end2 = np.array([np.cos(angle2), np.sin(angle2), 0])

        arrow2 = pv.Arrow(rot_grad_arrow_start, rot_grad_arrow_end, shaft_radius=0.03, tip_length=0.5, tip_radius=0.05)
        arrow3 = pv.Arrow(rot_grad_arrow_start2, rot_grad_arrow_end2, shaft_radius=0.03, tip_length=0.5, tip_radius=0.05)
        self.plotter.add_mesh(arrow2, color='pink')
        self.plotter.add_mesh(arrow3, color='pink')

    def vizu_image_masks(self, frame_idx, mask_orig, mask_loss=None):
        # Load your PNG image
        img_name = "{:06d}".format(frame_idx) + ".png"
        image_path = self.image_path + img_name
        image = plt.imread(image_path)

        # Get the dimensions of the image
        height, width, _ = image.shape

        # Create a figure with a size matching the image dimensions
        fig, ax = plt.subplots(figsize=(width / 100, height / 100))

        # Remove axis labels and ticks
        ax.axis('off')

        # Create a list of random colors for the masks
        mask_colors_gt = [(1., 0., 0.) for _ in mask_orig]
        mask_colors_det = [(0., 1., 0.) for _ in mask_loss]
        masked_image = image.copy()
        # Iterate through masks and overlay them with different colors and transparency
        for i, (car_mask, mask_color) in enumerate(zip(mask_orig, mask_colors_gt)):
            # Convert the boolean mask to a binary mask (0 or 1)
            car_mask = car_mask.cpu().detach().numpy().astype(np.uint8)

            # Create a masked image where the mask is applied with transparency
            masked_image[car_mask == 1] = (1 - 0.25) * masked_image[car_mask == 1] + 0.25 * np.array(mask_color)
        if mask_loss is not None:
            for i, (car_mask, mask_color) in enumerate(zip(mask_loss, mask_colors_det)):
                # Convert the boolean mask to a binary mask (0 or 1)
                car_mask = car_mask[0, :, :, 0].cpu().detach().numpy().astype(np.uint8)

                # Create a masked image where the mask is applied with transparency
                masked_image[car_mask == 1] = (1 - 0.25) * masked_image[car_mask == 1] + 0.25 * np.array(mask_color)
        # Display the plot with all masks
        ax.imshow(masked_image)
        plt.show()

    def vizu_image_masks_waymo(self, batch_dict, mask_orig, mask_loss=None, camera_idx=0):
        FILENAME = 'segment-' + batch_dict['metadata'][0]['context_name'] + '_with_camera_labels.tfrecord'
        FILENAME = '/home/potoso/openpcdet_modified/data/waymo/raw_data/' + FILENAME
        idx = int(batch_dict['sample_idx'].item())
        dataset = tf.data.TFRecordDataset(FILENAME, compression_type='')

        for i, data in enumerate(dataset):
            if i > idx:
                break
            if i != idx:
                continue
            frame = open_dataset.Frame()
            frame.ParseFromString(bytearray(data.numpy()))

            for index, image in enumerate(frame.images):
                if index == camera_idx:
                    img = tf.image.decode_jpeg(image.image)

        image = img.numpy()

        # Get the dimensions of the image
        height, width, _ = image.shape

        # Create a figure with a size matching the image dimensions
        fig, ax = plt.subplots(figsize=(width / 100, height / 100))

        # Remove axis labels and ticks
        ax.axis('off')

        # Create a list of random colors for the masks
        mask_colors_gt = [(255., 0., 0.) for _ in mask_orig]
        mask_colors_det = [(0., 255., 0.) for _ in mask_loss]
        masked_image = image.copy()
        # Iterate through masks and overlay them with different colors and transparency
        for i, (car_mask, mask_color) in enumerate(zip(mask_orig, mask_colors_gt)):
            # Convert the boolean mask to a binary mask (0 or 1)
            car_mask = car_mask.cpu().detach().numpy().astype(np.uint8)

            # Create a masked image where the mask is applied with transparency
            masked_image[car_mask == 1] = (1 - 0.25) * masked_image[car_mask == 1] + 0.25 * np.array(mask_color)
        if mask_loss is not None:
            for i, (car_mask, mask_color) in enumerate(zip(mask_loss, mask_colors_det)):
                # Convert the boolean mask to a binary mask (0 or 1)
                car_mask = car_mask[0, :, :, 0].cpu().detach().numpy().astype(np.uint8)

                # Create a masked image where the mask is applied with transparency
                masked_image[car_mask == 1] = (1 - 0.25) * masked_image[car_mask == 1] + 0.25 * np.array(mask_color)
        # Display the plot with all masks
        ax.imshow(masked_image)
        plt.show()


