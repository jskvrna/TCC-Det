import copy
import time
import pytorch3d.ops
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import LazyConfig, instantiate
from detectron2.engine.defaults import create_ddp_model
from pyntcloud import PyntCloud
from utils2 import load_velo_scan, get_perfect_scale
import scipy
from scipy.spatial.transform import Rotation as R
from scipy.ndimage import binary_erosion
from scipy.spatial.distance import cdist
import cv2, os
import pickle
import open3d
import torch
import zstd
import faiss
import tensorflow.compat.v1 as tf
from waymo_open_dataset.utils import frame_utils
from waymo_open_dataset import dataset_pb2 as open_dataset
from anno_V3 import AutoLabel3D
import numpy as np

class Loader(AutoLabel3D):
    def __init__(self, args):
        super().__init__(args)
        self.random_indexes = []
        self.mapping_data = []

        if self.args.dataset == 'waymo':
            with open(self.cfg.paths.waymo_info_path + "/train.txt", 'r') as f:
                self.random_indexes = [line.strip() for line in f.readlines()]
        else:
            with open(self.cfg.paths.kitti_path + 'object_detection/devkit_object/mapping/train_rand.txt', 'r') as f:
                line = f.readline().strip()
                self.random_indexes = line.split(',')

            with open(self.cfg.paths.kitti_path + 'object_detection/devkit_object/mapping/train_mapping.txt', 'r') as f:
                for line in f:
                    self.mapping_data.append(line.strip().split(' '))

    class Car:
        def __init__(self):
            self.lidar = None
            self.scale_lidar = None
            self.locations = None
            self.mask = None
            self.info = None
            self.moving = None
            self.img_index = None

            self.x = None
            self.y = None
            self.z = None
            self.theta = None
            self.length = None
            self.width = None
            self.height = None
            self.model = None
            self.optimized = False
            self.bbox = None

            self.x_scale = None
            self.y_scale = None
            self.z_scale = None
            self.theta_scale = None


    # Detectron
    def load_and_init_detectron_lazy(self):
        # Load the lazyconfig, we are using this, not the classic, because this regnety learned with different idea is used
        cfg = LazyConfig.load(self.cfg.paths.detectron_config)
        cfg = LazyConfig.apply_overrides(cfg, ['train.init_checkpoint=' + self.cfg.paths.model_path])

        # Init the model
        model = instantiate(cfg.model)
        # If we are using SAM, then locally we cannot fit it also with the detectron to memory
        if self.cfg.general.device == 'gpu':
            model.to(cfg.train.device)
        model = create_ddp_model(model)
        DetectionCheckpointer(model).load(cfg.train.init_checkpoint)
        model.eval()
        if not self.cfg.general.supress_debug_prints:
            print("Detectron2 Loaded")
        self.model = model

    def load_and_init_SAM(self):
        sam = sam_model_registry["vit_h"](checkpoint=self.cfg.paths.sam_path)
        if self.cfg.general.device == 'gpu':
            sam.to(device="cuda")
        self.sam_predictor = SamPredictor(sam)
        if not self.cfg.general.supress_debug_prints:
            print("SAM Loaded")

    def load_and_prepare_lidar_scan(self, filename, img):
        # First get all the lidar points
        lidar = np.array(load_velo_scan(self.cfg.paths.kitti_path + 'object_detection/training/velodyne/' + filename + '.bin'))
        self.prepare_scan(filename, img, lidar)
        self.prepare_img_dist(img)

    def load_and_prepare_lidar_scan_from_multiple_pykittiV2(self, filename, img, save=False):
        lidar_orig = np.array(load_velo_scan(self.cfg.paths.kitti_path + 'object_detection/training/velodyne/' + filename + '.bin'))

        if self.cfg.frames_creation.use_icp:
            transformations = self.calculate_transformationsV2(self.cfg.frames_creation.nscans_before, self.cfg.frames_creation.nscans_after)
        else:
            transformations = self.calculate_transformations(self.cfg.frames_creation.nscans_before, self.cfg.frames_creation.nscans_after)

        if not self.cfg.general.supress_debug_prints:
            print("Get standing car candidates")
        car_locations, car_locations_lidar, car_locations_masks = self.get_standing_car_candidates(transformations)

        if not self.cfg.general.supress_debug_prints:
            print("Perform 3D tracking")
        moving_cars, moving_cars_lidar, moving_cars_masks = self.perform_3D_tracking_kitti(car_locations, car_locations_lidar, car_locations_masks)

        if not self.cfg.general.supress_debug_prints:
            print("Create cars from features")
        cars = self.create_cars_from_extracted_feats_3DTrack(moving_cars, moving_cars_lidar, None, moving_cars_masks)

        if not self.cfg.general.supress_debug_prints:
            print("Decide if moving or standing car")
        cars = self.decide_if_standing_or_moving(cars, waymo=False)

        if not self.cfg.general.supress_debug_prints:
            print("Filter cars")
        cars = self.filter_moving_and_not_visible(cars, waymo=False)
        cars = self.choose_proper_mask(cars, waymo=False)

        if not self.cfg.general.supress_debug_prints:
            print("Final Modifications")
        cars = self.standing_concatenate_lidar(cars)
        cars = self.filter_hidden_standing_cars_tracked(cars, waymo=False)
        cars = self.moving_lidar_keep_ref(cars, waymo=False)

        for car in cars:
            if car.lidar is not None:
                # To reduce the size of the lidar in storage
                if len(car.lidar) > 10000:
                    car.lidar = self.downsample_random(car.lidar[:, :3], 10000)
                car.lidar = car.lidar.astype(np.float32)

        if save:
            compressed_arr = zstd.compress(pickle.dumps(cars, pickle.HIGHEST_PROTOCOL))

            if self.cfg.frames_creation.use_growing_for_point_extraction:
                with open(self.cfg.paths.merged_frames_path + "cars_3DTrack_growing/" + self.file_name + ".zstd", 'wb') as f:
                    f.write(compressed_arr)
            else:
                with open(self.cfg.paths.merged_frames_path + "cars_3DTrack/" + self.file_name + ".zstd", 'wb') as f:
                    f.write(compressed_arr)

        else:
            # If we do not save, which means we continue in optimizing, I want to remember the original scan for finding locations of the cars.
            self.lidar = self.prepare_scan(filename, img, lidar_orig, save=False, crop=not self.cfg.visualization.show_pcdet and not self.cfg.visualization.visu_whole_lidar)
            idx = 0
            for car in cars:
                idx += 1
                padding = np.ones((car.lidar.shape[0], 3))

                car.lidar = np.concatenate((car.lidar, padding), axis=1).T

            self.cars = sorted(cars, key=lambda x: x.lidar.shape[1], reverse=True)

    def load_merged_frames_from_files_KITTI(self):
        self.load_and_prepare_lidar_scan(self.file_name, self.img)

        if self.cfg.frames_creation.use_growing_for_point_extraction:
            with open(self.cfg.paths.merged_frames_path + "cars_3DTrack_growing/" + self.file_name + ".zstd", 'rb') as f:
                decompressed_data = zstd.decompress(f.read())
        else:
            with open(self.cfg.paths.merged_frames_path + "cars_3DTrack/" + self.file_name + ".zstd", 'rb') as f:
                decompressed_data = zstd.decompress(f.read())
        cars = pickle.loads(decompressed_data)
        self.cars = cars
        self.cars = sorted(self.cars, key=lambda x: x.lidar.shape[0], reverse=True)

        cur_lidar = np.ascontiguousarray(self.lidar.T[:, :3]).astype('float32')
        quantizer = faiss.IndexFlatL2(cur_lidar.shape[1])
        index_faiss = faiss.IndexIVFFlat(quantizer, cur_lidar.shape[1], int(np.floor(np.sqrt(cur_lidar.shape[0]))))
        index_faiss.train(cur_lidar)
        index_faiss.add(cur_lidar)
        index_faiss.nprobe = 10

        new_cars = []

        for car in self.cars:
            if car.lidar is not None:
                center = np.zeros((1, 3))
                center[0, 0] = np.median(car.lidar[:, 0])
                center[0, 1] = np.median(car.lidar[:, 1])
                center[0, 2] = np.median(car.lidar[:, 2])

                idx, distances, indexes = index_faiss.range_search(np.ascontiguousarray(center).astype('float32'),
                                                                   2. ** 2)

                if len(distances) < 1:
                    continue

                padding = np.ones((car.lidar.shape[0], 3))

                car.lidar = np.concatenate((car.lidar, padding), axis=1).T
                new_cars.append(car)
        self.cars = new_cars

    def load_and_prepare_lidar_scan_from_multiple_waymo(self, save=False):

        transformations = self.calculate_transformations_waymo(self.cfg.frames_creation.nscans_before, self.cfg.frames_creation.nscans_after)

        if not self.cfg.general.supress_debug_prints:
            print("Convert to current frame")
        car_locations, car_locations_lidar, car_locations_info, car_locations_masks, detectron_output_arr = self.convert_to_current_frame(transformations)

        if not self.cfg.general.supress_debug_prints:
            print("Perform 3D tracking")
        moving_cars, moving_cars_lidar, moving_cars_info, moving_cars_masks = self.perform_3D_tracking(car_locations, car_locations_lidar, car_locations_info, car_locations_masks)

        if not self.cfg.general.supress_debug_prints:
            print("Create cars from features")
        cars = self.create_cars_from_extracted_feats_3DTrack(moving_cars, moving_cars_lidar, moving_cars_info, moving_cars_masks)

        # Delete all moving cars which cannot be seen in the reference frames, thus probably not labeled.
        if not self.cfg.general.supress_debug_prints:
            print("Decide if moving or standing car")
        cars = self.decide_if_standing_or_moving(cars)
        #cars = self.decide_if_moving(cars)

        if not self.cfg.general.supress_debug_prints:
            print("Filter cars")
        # Remove all moving outside of the ref frame
        cars = self.filter_moving_and_not_visible(cars)
        cars = self.choose_proper_mask(cars)

        if not self.cfg.general.supress_debug_prints:
            print("Final Modifications")
        cars = self.standing_concatenate_lidar(cars)
        cars = self.filter_hidden_standing_cars_tracked(cars)
        cars = self.moving_lidar_keep_ref(cars)

        for car in cars:
            if car.lidar is not None:
                #To reduce the size of the lidar in storage
                if len(car.lidar) > 10000:
                    car.lidar = self.downsample_random(car.lidar[:, :3], 10000)
                car.lidar = car.lidar.astype(np.float32)

        if save:
            compressed_arr = zstd.compress(pickle.dumps(cars, pickle.HIGHEST_PROTOCOL))

            if self.cfg.frames_creation.use_growing_for_point_extraction:
                if not os.path.isdir(self.cfg.paths.merged_frames_path + "cars_3DTrack_growing/" + self.file_name):
                    os.mkdir(self.cfg.paths.merged_frames_path + "cars_3DTrack_growing/" + self.file_name)
                with open(self.cfg.paths.merged_frames_path + "cars_3DTrack_growing/" + self.file_name + "/" + str(self.pic_index) + ".zstd", 'wb') as f:
                    f.write(compressed_arr)
            else:
                if not os.path.isdir(self.cfg.paths.merged_frames_path + "cars_3DTrack/" + self.file_name):
                    os.mkdir(self.cfg.paths.merged_frames_path + "cars_3DTrack/" + self.file_name)
                with open(self.cfg.paths.merged_frames_path + "cars_3DTrack/" + self.file_name + "/" + str(self.pic_index) + ".zstd",'wb') as f:
                    f.write(compressed_arr)

        else:
            # If we do not save, which means we continue in optimizing, I want to remember the original scan for finding locations of the cars.
            self.lidar = self.waymo_lidar[self.pic_index].T
            idx = 0
            for car in cars:
                idx += 1
                padding = np.ones((car.lidar.shape[0], 3))

                car.lidar = np.concatenate((car.lidar, padding), axis=1).T

            self.cars = sorted(cars, key=lambda x: x.lidar.shape[1], reverse=True)

    def load_and_prepare_lidar_scan_from_multiple_waymo_tracker(self, save=False):
        # The reference frame - from IMU to World
        if not self.cfg.general.supress_debug_prints:
            print("Calculating transformations")
        transformations = self.calculate_transformations_waymo(self.cfg.frames_creation.nscans_before, self.cfg.frames_creation.nscans_after)

        if not self.cfg.general.supress_debug_prints:
            print("Create cars from extracted features")
        cars = self.create_cars_from_extracted_feats()

        if not self.cfg.general.supress_debug_prints:
            print("Transform points into reference frame")
        #Transform all points into this reference frame
        cars = self.transform_lidar_points_to_reference_frame(cars, transformations)
        cars = self.transform_lidar_positions_to_reference_frame(cars, transformations)

        if not self.cfg.general.supress_debug_prints:
            print("Decide if standing or moving")
        #Now we need to decide if they are moving or not.
        cars = self.decide_if_standing_or_moving(cars)
        #cars = self.decide_if_moving(cars)

        if not self.cfg.general.supress_debug_prints:
            print("Remove moving cars which are not visible")
        #Remove all moving outside of the ref frame
        cars = self.filter_moving_and_not_visible(cars)

        if not self.cfg.general.supress_debug_prints:
            print("Final modifications")
        #Now we have all we need. For standing concat the lidar, for moving, keep only the ref one
        cars = self.standing_concatenate_lidar(cars)
        cars = self.filter_hidden_standing_cars_tracked(cars)
        cars = self.moving_lidar_keep_ref(cars)

        for car in cars:
            if car.lidar is not None:
                #To reduce the size of the lidar in storage
                if len(car.lidar) > 10000:
                    car.lidar = self.downsample_random(car.lidar[:, :3], 10000)
                car.lidar = car.lidar.astype(np.float32)

        # We want to save the results
        if save:
            compressed_arr = zstd.compress(pickle.dumps(cars, pickle.HIGHEST_PROTOCOL))

            if self.cfg.frames_creation.use_growing_for_point_extraction:
                if not os.path.isdir(self.cfg.paths.merged_frames_path + "cars_2DTrack_growing/" + self.file_name):
                    os.mkdir(self.cfg.paths.merged_frames_path + "cars_2DTrack_growing/" + self.file_name)

                with open(self.cfg.paths.merged_frames_path + "cars_2DTrack_growing/" + self.file_name + "/" + str(self.pic_index) + ".zstd", 'wb') as f:
                    f.write(compressed_arr)
            else:
                if not os.path.isdir(self.cfg.paths.merged_frames_path + "cars_2DTrack/" + self.file_name):
                    os.mkdir(self.cfg.paths.merged_frames_path + "cars_2DTrack/" + self.file_name)
                with open(self.cfg.paths.merged_frames_path + "cars_2DTrack/" + self.file_name + "/" + str(self.pic_index) + ".zstd", 'wb') as f:
                    f.write(compressed_arr)
        else:
            # If we do not save, which means we continue in optimizing, I want to remember the original scan for finding locations of the cars.
            self.lidar = self.waymo_lidar[self.pic_index].T
            for car in cars:
                padding = np.ones((car.lidar.shape[0], 3))

                car.lidar = np.concatenate((car.lidar, padding), axis=1).T

            self.cars = sorted(cars, key=lambda x: x.lidar.shape[1], reverse=True)

    def choose_proper_mask(self, cars, waymo=True):
        for car in cars:
            hidden = True
            for z in range(len(car.locations)):
                if car.locations[z] is not None:
                    if waymo:
                        frame_idx = car.info[z][1]
                        if self.pic_index == frame_idx:
                            car.mask = car.mask[z]
                            car.img_index = car.info[z][2]
                            hidden = False
                            break
                    else:
                        frame_idx = car.locations[z][3]
                        if frame_idx == 0:
                            car.mask = car.mask[z]
                            hidden = False
                            break
            if hidden:
                car.mask = None
        return cars

    def create_cars_from_extracted_feats(self):
        cars = []
        for i in range(len(self.compressed_detandtracked)):
            car = self.Car()
            decompressed_data = zstd.decompress(self.compressed_detandtracked[i])
            to_load = pickle.loads(decompressed_data)

            car.lidar = to_load.lidar_points
            car.locations = to_load.lidar_locations
            car.info = to_load.lidar_info

            for z in range(len(car.info)):
                if car.info[z] is not None:
                    if car.info[z][1] == self.pic_index:
                        tmp_mask = to_load.masks[z]
                        tmp_img_index = car.info[z][2]
                        if tmp_mask is not None:
                            # Now convert the stitched img back to "normal" one
                            # TODO, take both masks and utilize them...
                            car.mask, car.img_index = self.convert_stitched_img_to_normal(tmp_mask, tmp_img_index)
                        break

            cars.append(car)

        return cars

    def create_cars_from_extracted_feats_3DTrack(self, moving_cars, moving_cars_lidar, moving_cars_info, moving_cars_masks):
        cars = []
        for i in range(len(moving_cars)):
            car = self.Car()
            car.lidar = moving_cars_lidar[i]
            car.locations = moving_cars[i]
            if moving_cars_info is not None:
                car.info = moving_cars_info[i]
            car.mask = moving_cars_masks[i]

            cars.append(car)

        return cars
    def convert_stitched_img_to_normal(self, mask, img_index):
        # Inverse of the mask
        if img_index == 0:
            if self.cfg.general.device == 'cpu':
                img0, img1 = self.inverse_of_mask_img01(mask.to(torch.float32), self.homos_all[img_index])
            else:
                img0, img1 = self.inverse_of_mask_img01(mask.cuda().to(torch.float32), self.homos_all[img_index])
            img0 = img0.numpy()[-886:, :]
            img1 = img1.numpy()

            if np.sum(img0) > np.sum(img1):
                return img0, 4
            else:
                return img1, 2

        elif img_index == 1:
            if self.cfg.general.device == 'cpu':
                img1, img2 = self.inverse_of_mask_img01(mask.to(torch.float32), self.homos_all[img_index])
            else:
                img1, img2 = self.inverse_of_mask_img01(mask.cuda().to(torch.float32), self.homos_all[img_index])
            img1 = img1.numpy()
            img2 = img2.numpy()

            if np.sum(img1) > np.sum(img2):
                return img1, 2
            else:
                return img2, 1


        elif img_index == 2:
            if self.cfg.general.device == 'cpu':
                img3, img2 = self.inverse_of_mask_img23(mask.to(torch.float32), self.homos_all[img_index])
            else:
                img3, img2 = self.inverse_of_mask_img23(mask.cuda().to(torch.float32), self.homos_all[img_index])
            img2 = img2.numpy()
            img3 = img3.numpy()

            if np.sum(img2) > np.sum(img3):
                return img2, 1
            else:
                return img3, 3

        else:
            if self.cfg.general.device == 'cpu':
                img4, img3 = self.inverse_of_mask_img23(mask.to(torch.float32), self.homos_all[img_index])
            else:
                img4, img3 = self.inverse_of_mask_img23(mask.cuda().to(torch.float32), self.homos_all[img_index])
            img3 = img3.numpy()
            img4 = img4.numpy()[-886:, :]

            if np.sum(img3) > np.sum(img4):
                return img3, 3
            else:
                return img4, 5

    def moving_lidar_keep_ref(self, cars, waymo=True):
        for car in cars:
            if car.moving:
                for z in range(len(car.locations)):
                    if car.locations[z] is not None:
                        if waymo:
                            frame_idx = car.info[z][1]
                            if self.pic_index == frame_idx:
                                car.lidar = car.lidar[z]
                                break
                        else:
                            frame_idx = car.locations[z][3]
                            if frame_idx == 0:
                                car.lidar = car.lidar[z]
                                break
        return cars

    def standing_concatenate_lidar(self, cars):
        for car in cars:
            if len(car.lidar) > 0 and not car.moving:
                tmp_lidar = [arr for arr in car.lidar if arr is not None]
                if len(tmp_lidar) > 0:
                    car.lidar = np.concatenate(tmp_lidar, axis=0)
                else:
                    car.lidar = None
        return cars

    def decide_if_moving(self, cars):
        for car in cars:
            dists = []
            for z in range(len(car.locations) - 1):
                if car.locations[z] is not None and car.locations[z + 1] is not None:
                    dists.append(np.linalg.norm(car.locations[z][:2] - car.locations[z + 1][:2]))
            median = np.median(dists)
            if np.abs(median) > 2. / 10.:
                car.moving = True
            else:
                car.moving = False

        return cars

    def filter_moving_and_not_visible(self, cars, waymo=True):
        out_cars = []

        for car in cars:
            moving_and_hidden = True
            if car.moving:
                for z in range(len(car.locations)):
                    if car.locations[z] is not None:
                        if waymo:
                            frame_idx = car.info[z][1]
                            if self.pic_index == frame_idx:
                                moving_and_hidden = False
                                break
                        else:
                            frame_idx = car.locations[z][3]
                            if frame_idx == 0:
                                moving_and_hidden = False
                                break
                if not moving_and_hidden:
                    out_cars.append(car)
            else:
                out_cars.append(car)

        return out_cars

    def filter_hidden_standing_cars_tracked(self, cars, waymo=True):
        out_cars = []

        # First build the faiss index of lidar
        if waymo:
            cur_lidar = self.waymo_lidar[self.pic_index][:, :3]
        else:
            cur_lidar = np.array(load_velo_scan(self.cfg.paths.kitti_path + 'object_detection/training/velodyne/' + self.file_name + '.bin'))[:, :3]
        index = self.create_faiss_tree(cur_lidar)

        for car in cars:
            if not car.moving:
                if car.lidar is not None:
                    idx, distances, indexes = index.range_search(np.ascontiguousarray(car.lidar).astype('float32'), 0.1 ** 2)
                    if len(idx) > 0:
                        out_cars.append(car)
            else:
                out_cars.append(car)

        return out_cars

    def transform_lidar_points_to_reference_frame(self, cars, transformations):
        for car in cars:
            for z in range(len(car.lidar)):
                if car.info[z] is not None and car.lidar[z] is not None:
                    frame_idx = car.info[z][1]
                    if np.abs(self.pic_index - frame_idx) > self.cfg.frames_creation.nscans_before:
                        car.lidar[z] = None
                    elif self.pic_index - frame_idx < 0 or self.pic_index - frame_idx > 0:
                        car.lidar[z] = np.matmul(transformations[self.cfg.frames_creation.nscans_before - (self.pic_index - frame_idx)][0:3, 0:3], car.lidar[z].T).T
                        car.lidar[z] += transformations[self.cfg.frames_creation.nscans_before - (self.pic_index - frame_idx)][0:3, 3]
        return cars

    def transform_lidar_positions_to_reference_frame(self, cars, transformations):
        for car in cars:
            for z in range(len(car.locations)):
                if car.info[z] is not None and car.locations[z] is not None:
                    frame_idx = car.info[z][1]
                    if np.abs(self.pic_index - frame_idx) > self.cfg.frames_creation.nscans_before:
                        car.locations[z] = None
                    elif self.pic_index - frame_idx < 0 or self.pic_index - frame_idx > 0:
                        car.locations[z] = np.matmul(transformations[self.cfg.frames_creation.nscans_before - (self.pic_index - frame_idx)][0:3, 0:3], car.locations[z].T).T
                        car.locations[z] += transformations[self.cfg.frames_creation.nscans_before - (self.pic_index - frame_idx)][0:3, 3]
        return cars

    def calculate_transformations(self, nscans_before, nscans_after, save=False):
        if self.load_merged_frames or self.load_transformations:
            transformations = np.load(self.cfg.paths.merged_frames_path + "transformations/" + self.file_name + '.npy')
            # We know that the transformation were generated for -50 and 120 frames.
            transformations = transformations[self.cfg.frames_creation.nscans_transformation_range - nscans_before: self.cfg.frames_creation.nscans_transformation_range + 1 + nscans_after]
        else:
            T_w_imu_ref = self.kitti_data.oxts[self.file_number].T_w_imu
            num_of_transformations = nscans_before + nscans_after + 1  # +1 because we actually want 1 more
            transformations = np.zeros((num_of_transformations, 4, 4))

            path_to_ref_velo = self.path_to_folder + 'velodyne_points/data/' + f'{self.file_number :0>10}' + '.bin'
            if self.cfg.frames_creation.use_icp:
                lidar_ref = np.array(load_velo_scan(path_to_ref_velo))
                lidar_ref = self.transform_velo_to_cam(self.file_name, lidar_ref, filter_points=False)
                lidar_ref = lidar_ref[:3, :].T

            for i in range(-nscans_before, nscans_after + 1):
                path_to_cur_velo = self.path_to_folder + 'velodyne_points/data/' + f'{self.file_number + i :0>10}' + '.bin'
                #Check that we have everything we need
                if self.file_number + i < 0 or self.file_number + i >= len(self.kitti_data.oxts) or not os.path.exists(
                        path_to_cur_velo):
                    continue

                # Load cur frame - IMU to world
                T_w_imu_cur = self.kitti_data.oxts[self.file_number + i].T_w_imu

                # Compute the transformation between frames - IMU cur to world then to IMU again but to the ref frame
                T_cur_to_ref = np.matmul(np.linalg.inv(T_w_imu_ref), T_w_imu_cur)
                # Now we need to go from IMU to CAM2
                T_imu_to_cam = self.kitti_data.calib.T_cam2_imu

                T_cur_to_ref = np.matmul(T_cur_to_ref, np.linalg.inv(T_imu_to_cam))
                T_cur_to_ref = np.matmul(T_imu_to_cam, T_cur_to_ref)

                if self.cfg.frames_creation.use_icp:
                    # Now lets try the ICP
                    # Load the velo scan
                    lidar_cur = np.array(load_velo_scan(path_to_cur_velo))
                    lidar_cur = self.transform_velo_to_cam(self.file_name, lidar_cur, filter_points=False)
                    lidar_cur = lidar_cur[:3, :]
                    # Transform the points between frames.
                    lidar_cur = np.matmul(T_cur_to_ref[0:3, 0:3], lidar_cur).T
                    lidar_cur += T_cur_to_ref[0:3, 3]

                    new_transformation = self.icp_point_to_plane_open3d(lidar_cur, lidar_ref)
                    new_transformation = copy.deepcopy(new_transformation)
                    rot = R.from_matrix(new_transformation[:3, :3])
                    T_cur_to_ref = np.matmul(new_transformation, T_cur_to_ref)

                transformations[i + nscans_before, :, :] = T_cur_to_ref

        if save:
            np.save(self.cfg.paths.merged_frames_path + "transformations/" + self.file_name, transformations)
        else:
            return transformations

    #Always works with ICP
    def calculate_transformationsV2(self, nscans_before, nscans_after, save=False):
        if self.load_merged_frames or self.load_transformations:
            transformations = np.load(self.cfg.paths.merged_frames_path + "transformations/" + self.file_name + '.npy')
            transformations = transformations[self.cfg.frames_creation.nscans_transformation_range - nscans_before: self.cfg.frames_creation.nscans_transformation_range + 1 + nscans_after]
        else:
            num_of_transformations = nscans_before + nscans_after + 1  # +1 because we actually want 1 more
            transformations = np.zeros((num_of_transformations, 4, 4))
            transformations[nscans_before, :, :] = np.eye(4)
            tmp_transformations = np.zeros((num_of_transformations, 4, 4))

            jump_step = 5

            for i in range(tmp_transformations.shape[0]):
                tmp_transformations[i, :, :] = np.eye(4)
                transformations[i, :, :] = np.eye(4)

            if not self.cfg.general.supress_debug_prints:
                print("Calculating transformations with step")
            for i in range(-nscans_before, nscans_after + 1, jump_step):
                if i == 0:
                    continue

                if i < 0:
                    path_to_cur_velo = self.path_to_folder + 'velodyne_points/data/' + f'{self.file_number + i :0>10}' + '.bin'
                    path_to_ref_velo = self.path_to_folder + 'velodyne_points/data/' + f'{self.file_number + i + jump_step:0>10}' + '.bin'
                else:
                    path_to_ref_velo = self.path_to_folder + 'velodyne_points/data/' + f'{self.file_number + i - jump_step:0>10}' + '.bin'
                    path_to_cur_velo = self.path_to_folder + 'velodyne_points/data/' + f'{self.file_number + i :0>10}' + '.bin'

                if self.file_number + i < 0 or self.file_number + i >= len(self.kitti_data.oxts) or not os.path.exists(
                        path_to_cur_velo) or not os.path.exists(path_to_ref_velo):
                    if i < 0 and i + jump_step * 2 <= 0:
                        path_to_cur_velo = self.path_to_folder + 'velodyne_points/data/' + f'{self.file_number + i :0>10}' + '.bin'
                        path_to_ref_velo = self.path_to_folder + 'velodyne_points/data/' + f'{self.file_number + i + jump_step * 2:0>10}' + '.bin'
                    elif i >= 0 and i - jump_step * 2 >= 0:
                        path_to_ref_velo = self.path_to_folder + 'velodyne_points/data/' + f'{self.file_number + i - jump_step * 2 :0>10}' + '.bin'
                        path_to_cur_velo = self.path_to_folder + 'velodyne_points/data/' + f'{self.file_number + i :0>10}' + '.bin'
                    else:
                        continue

                    if self.file_number + i < 0 or self.file_number + i >= len(
                            self.kitti_data.oxts) or not os.path.exists(
                            path_to_cur_velo) or not os.path.exists(path_to_ref_velo):
                        if i < 0 and i + jump_step * 3 <= 0:
                            path_to_cur_velo = self.path_to_folder + 'velodyne_points/data/' + f'{self.file_number + i :0>10}' + '.bin'
                            path_to_ref_velo = self.path_to_folder + 'velodyne_points/data/' + f'{self.file_number + i + jump_step * 3:0>10}' + '.bin'
                        elif i >= 0 and i - jump_step * 3 >= 0:
                            path_to_ref_velo = self.path_to_folder + 'velodyne_points/data/' + f'{self.file_number + i - jump_step * 3:0>10}' + '.bin'
                            path_to_cur_velo = self.path_to_folder + 'velodyne_points/data/' + f'{self.file_number + i :0>10}' + '.bin'
                        else:
                            continue

                        if self.file_number + i < 0 or self.file_number + i >= len(
                                self.kitti_data.oxts) or not os.path.exists(
                            path_to_cur_velo) or not os.path.exists(path_to_ref_velo):
                            if i < 0 and i + jump_step * 3 <= 0:
                                path_to_cur_velo = self.path_to_folder + 'velodyne_points/data/' + f'{self.file_number + i :0>10}' + '.bin'
                                path_to_ref_velo = self.path_to_folder + 'velodyne_points/data/' + f'{self.file_number + i + jump_step * 4:0>10}' + '.bin'
                            elif i >= 0 and i - jump_step * 3 >= 0:
                                path_to_ref_velo = self.path_to_folder + 'velodyne_points/data/' + f'{self.file_number + i - jump_step * 4:0>10}' + '.bin'
                                path_to_cur_velo = self.path_to_folder + 'velodyne_points/data/' + f'{self.file_number + i :0>10}' + '.bin'
                            else:
                                continue

                            if self.file_number + i < 0 or self.file_number + i >= len(
                                    self.kitti_data.oxts) or not os.path.exists(
                                path_to_cur_velo) or not os.path.exists(path_to_ref_velo):
                                if i < 0 and i + jump_step * 3 <= 0:
                                    path_to_cur_velo = self.path_to_folder + 'velodyne_points/data/' + f'{self.file_number + i :0>10}' + '.bin'
                                    path_to_ref_velo = self.path_to_folder + 'velodyne_points/data/' + f'{self.file_number + i + jump_step * 5:0>10}' + '.bin'
                                elif i >= 0 and i - jump_step * 3 >= 0:
                                    path_to_ref_velo = self.path_to_folder + 'velodyne_points/data/' + f'{self.file_number + i - jump_step * 5:0>10}' + '.bin'
                                    path_to_cur_velo = self.path_to_folder + 'velodyne_points/data/' + f'{self.file_number + i :0>10}' + '.bin'
                                else:
                                    continue

                                if self.file_number + i < 0 or self.file_number + i >= len(
                                        self.kitti_data.oxts) or not os.path.exists(
                                    path_to_cur_velo) or not os.path.exists(path_to_ref_velo):
                                    continue
                                else:
                                    if i < 0:
                                        T_w_imu_cur = self.kitti_data.oxts[self.file_number + i].T_w_imu
                                        T_w_imu_ref = self.kitti_data.oxts[self.file_number + i + jump_step * 5].T_w_imu
                                    else:
                                        T_w_imu_ref = self.kitti_data.oxts[self.file_number + i - jump_step * 5].T_w_imu
                                        T_w_imu_cur = self.kitti_data.oxts[self.file_number + i].T_w_imu
                            else:
                                if i < 0:
                                    T_w_imu_cur = self.kitti_data.oxts[self.file_number + i].T_w_imu
                                    T_w_imu_ref = self.kitti_data.oxts[self.file_number + i + jump_step * 4].T_w_imu
                                else:
                                    T_w_imu_ref = self.kitti_data.oxts[self.file_number + i - jump_step * 4].T_w_imu
                                    T_w_imu_cur = self.kitti_data.oxts[self.file_number + i].T_w_imu
                        else:
                            if i < 0:
                                T_w_imu_cur = self.kitti_data.oxts[self.file_number + i].T_w_imu
                                T_w_imu_ref = self.kitti_data.oxts[self.file_number + i + jump_step * 3].T_w_imu
                            else:
                                T_w_imu_ref = self.kitti_data.oxts[self.file_number + i - jump_step * 3].T_w_imu
                                T_w_imu_cur = self.kitti_data.oxts[self.file_number + i].T_w_imu
                    else:
                        if i < 0:
                            T_w_imu_cur = self.kitti_data.oxts[self.file_number + i].T_w_imu
                            T_w_imu_ref = self.kitti_data.oxts[self.file_number + i + jump_step * 2].T_w_imu
                        else:
                            T_w_imu_ref = self.kitti_data.oxts[self.file_number + i - jump_step * 2].T_w_imu
                            T_w_imu_cur = self.kitti_data.oxts[self.file_number + i].T_w_imu
                else:
                    if i < 0:
                        T_w_imu_cur = self.kitti_data.oxts[self.file_number + i].T_w_imu
                        T_w_imu_ref = self.kitti_data.oxts[self.file_number + i + jump_step].T_w_imu
                    else:
                        T_w_imu_ref = self.kitti_data.oxts[self.file_number + i - jump_step].T_w_imu
                        T_w_imu_cur = self.kitti_data.oxts[self.file_number + i].T_w_imu

                lidar_ref = np.array(load_velo_scan(path_to_ref_velo))
                lidar_ref = self.transform_velo_to_cam(self.file_name, lidar_ref, filter_points=False)
                lidar_ref = lidar_ref[:3, :].T

                # Compute the transformation between frames - IMU cur to world then to IMU again but to the ref frame
                T_cur_to_ref = np.matmul(np.linalg.inv(T_w_imu_ref), T_w_imu_cur)
                # Now we need to go from IMU to CAM2
                T_imu_to_cam = self.kitti_data.calib.T_cam2_imu

                T_cur_to_ref = np.matmul(T_cur_to_ref, np.linalg.inv(T_imu_to_cam))
                T_cur_to_ref = np.matmul(T_imu_to_cam, T_cur_to_ref)

                lidar_cur = np.array(load_velo_scan(path_to_cur_velo))
                lidar_cur = self.transform_velo_to_cam(self.file_name, lidar_cur, filter_points=False)
                lidar_cur = lidar_cur[:3, :]
                # Transform the points between frames.
                lidar_cur = np.matmul(T_cur_to_ref[0:3, 0:3], lidar_cur).T
                lidar_cur += T_cur_to_ref[0:3, 3]

                new_transformation = self.icp_point_to_plane_open3d(lidar_cur, lidar_ref)
                new_transformation = copy.deepcopy(new_transformation)
                rot = R.from_matrix(new_transformation[:3, :3])
                T_cur_to_ref = np.matmul(new_transformation, T_cur_to_ref)

                tmp_transformations[i + nscans_before, :, :] = T_cur_to_ref

            if not self.cfg.general.supress_debug_prints:
                print("Calculating absolute transformations with step")
            for i in range(-nscans_before, 0, jump_step):
                path_to_cur_velo = self.path_to_folder + 'velodyne_points/data/' + f'{self.file_number + i :0>10}' + '.bin'

                if self.file_number + i < 0 or self.file_number + i >= len(self.kitti_data.oxts) or not os.path.exists(
                        path_to_cur_velo):
                    continue

                t_matrix = np.eye(4)
                for z in range(i, 0, jump_step):
                    t_matrix = np.matmul(tmp_transformations[z + nscans_before, :, :], t_matrix)
                transformations[i + nscans_before, :, :] = t_matrix

            for i in range(nscans_after, 0, -jump_step):
                path_to_cur_velo = self.path_to_folder + 'velodyne_points/data/' + f'{self.file_number + i:0>10}' + '.bin'

                if self.file_number + i < 0 or self.file_number + i >= len(self.kitti_data.oxts) or not os.path.exists(
                        path_to_cur_velo):
                    continue

                t_matrix = np.eye(4)
                for z in range(i, 0, -jump_step):
                    t_matrix = np.matmul(tmp_transformations[z + nscans_before, :, :], t_matrix)
                transformations[i + nscans_before, :, :] = t_matrix

            if not self.cfg.general.supress_debug_prints:
                print("Calculating absolute transformations for all frames")
            for i in range(-nscans_before, nscans_after + 1):
                if i % jump_step == 0:
                    continue
                if i < 0:
                    ref_index = int(np.rint(np.ceil(i/float(jump_step))*jump_step))
                else:
                    ref_index = int(np.rint(np.floor(i/float(jump_step))*jump_step))

                if np.array_equal(transformations[nscans_before + ref_index], np.eye(4)) and ref_index != 0:
                    if i < 0:
                        ref_index = int(np.rint(np.ceil(i / float(jump_step * 2)) * jump_step * 2))
                    else:
                        ref_index = int(np.rint(np.floor(i / float(jump_step * 2)) * jump_step * 2))

                if np.array_equal(transformations[nscans_before + ref_index], np.eye(4)) and ref_index != 0:
                    if i < 0:
                        ref_index = int(np.rint(np.ceil(i / float(jump_step * 3)) * jump_step * 3))
                    else:
                        ref_index = int(np.rint(np.floor(i / float(jump_step * 3)) * jump_step * 3))

                path_to_cur_velo = self.path_to_folder + 'velodyne_points/data/' + f'{self.file_number + i :0>10}' + '.bin'
                path_to_ref_velo = self.path_to_folder + 'velodyne_points/data/' + f'{self.file_number + ref_index :0>10}' + '.bin'

                if self.file_number + i < 0 or self.file_number + i >= len(self.kitti_data.oxts) or not os.path.exists(
                        path_to_cur_velo) or not os.path.exists(path_to_ref_velo):
                    continue

                T_w_imu_cur = self.kitti_data.oxts[self.file_number + i].T_w_imu
                T_w_imu_ref = self.kitti_data.oxts[self.file_number + ref_index].T_w_imu

                lidar_ref = np.array(load_velo_scan(path_to_ref_velo))
                lidar_ref = self.transform_velo_to_cam(self.file_name, lidar_ref, filter_points=False)
                lidar_ref = lidar_ref[:3, :].T

                # Compute the transformation between frames - IMU cur to world then to IMU again but to the ref frame
                T_cur_to_ref = np.matmul(np.linalg.inv(T_w_imu_ref), T_w_imu_cur)
                # Now we need to go from IMU to CAM2
                T_imu_to_cam = self.kitti_data.calib.T_cam2_imu

                T_cur_to_ref = np.matmul(T_cur_to_ref, np.linalg.inv(T_imu_to_cam))
                T_cur_to_ref = np.matmul(T_imu_to_cam, T_cur_to_ref)

                lidar_cur = np.array(load_velo_scan(path_to_cur_velo))
                lidar_cur = self.transform_velo_to_cam(self.file_name, lidar_cur, filter_points=False)
                lidar_cur = lidar_cur[:3, :]
                # Transform the points between frames.
                lidar_cur = np.matmul(T_cur_to_ref[0:3, 0:3], lidar_cur).T
                lidar_cur += T_cur_to_ref[0:3, 3]

                new_transformation = self.icp_point_to_plane_open3d(lidar_cur, lidar_ref)
                new_transformation = copy.deepcopy(new_transformation)
                rot = R.from_matrix(new_transformation[:3, :3])
                T_cur_to_ref = np.matmul(new_transformation, T_cur_to_ref)

                transformations[i + nscans_before, :, :] = np.matmul(transformations[nscans_before + ref_index], T_cur_to_ref)

        if save:
            np.save(self.cfg.paths.merged_frames_path + "transformations/" + self.file_name, transformations)
        else:
            return transformations

    def calculate_transformations_waymo(self, nscans_before, nscans_after, save=False):
        if self.load_merged_frames or self.load_transformations:
            transformations = np.load(self.cfg.paths.merged_frames_path + "transformations/" + self.file_name + "/" + str(self.pic_index) + '.npy')
            # We know that the transformation were generated for -50 and 120 frames.
            transformations = transformations[self.cfg.frames_creation.nscans_transformation_range - nscans_before: self.cfg.frames_creation.nscans_transformation_range + 1 + nscans_after]
        else:
            if os.path.isfile(self.cfg.paths.merged_frames_path + "/transformations/" + self.file_name + "/" + str(self.pic_index) + '.npy'):
                return
            num_of_transformations = nscans_before + nscans_after + 1  # +1 because we actually want 1 more
            transformations = np.zeros((num_of_transformations, 4, 4))
            transformations[nscans_before, :, :] = np.eye(4)
            tmp_transformations = np.zeros((num_of_transformations, 4, 4))

            for i in range(tmp_transformations.shape[0]):
                tmp_transformations[i, :, :] = np.eye(4)
                transformations[i, :, :] = np.eye(4)

            if not self.cfg.general.supress_debug_prints:
                print("Calculating transformations with step")
            for i in range(-nscans_before, nscans_after + 1):
                if i == 0:
                    continue

                transformation = self.get_transformation_icp(i)
                if transformation is not None:
                    tmp_transformations[i + nscans_before, :, :] = transformation
                    transformations[i + nscans_before, :, :] = transformation

            if not self.cfg.general.supress_debug_prints:
                print("Calculating absolute transformations with step")
            for i in range(-nscans_before, 0):
                if i + self.pic_index < 0:
                    continue

                t_matrix = np.eye(4)
                for z in range(i, 0):
                    t_matrix = np.matmul(tmp_transformations[z + nscans_before, :, :], t_matrix)
                transformations[i + nscans_before, :, :] = t_matrix

            for i in range(nscans_after, 0, -1):
                if i + self.pic_index >= len(self.waymo_data):
                    continue

                t_matrix = np.eye(4)
                for z in range(i, 0, -1):
                    t_matrix = np.matmul(tmp_transformations[z + nscans_before, :, :], t_matrix)
                transformations[i + nscans_before, :, :] = t_matrix
        if save:
            if not os.path.isdir(self.cfg.paths.merged_frames_path + "transformations/" + self.file_name):
                os.mkdir(self.cfg.paths.merged_frames_path + "transformations/" + self.file_name)
            np.save(self.cfg.paths.merged_frames_path + "transformations/" + self.file_name + "/" + str(self.pic_index), transformations)
        else:
            return transformations

    def get_transformation_icp(self, i):
        if i < 0:
            if i + self.pic_index < 0:
                return None
            frame_cur = self.waymo_frame[self.pic_index + i]
            frame_ref = self.waymo_frame[self.pic_index + i + 1]

            T_w_imu_cur = np.array(frame_cur.pose.transform).reshape((4, 4))
            T_w_imu_ref = np.array(frame_ref.pose.transform).reshape((4, 4))

            lidar_cur = self.waymo_lidar[self.pic_index + i][:, :3]
            lidar_ref = self.waymo_lidar[self.pic_index + i + 1][:, :3]

        else:
            if i + self.pic_index >= len(self.waymo_data):
                return None
            frame_cur = self.waymo_frame[self.pic_index + i]
            frame_ref = self.waymo_frame[self.pic_index + i - 1]

            T_w_imu_cur = np.array(frame_cur.pose.transform).reshape((4, 4))
            T_w_imu_ref = np.array(frame_ref.pose.transform).reshape((4, 4))

            lidar_cur = self.waymo_lidar[self.pic_index + i][:, :3]
            lidar_ref = self.waymo_lidar[self.pic_index + i - 1][:, :3]

        # Compute the transformation between frames - IMU cur to world then to IMU again but to the ref frame
        T_cur_to_ref = np.matmul(np.linalg.inv(T_w_imu_ref), T_w_imu_cur)

        # Transform the points between frames.
        lidar_cur_tmp = np.matmul(T_cur_to_ref[0:3, 0:3], lidar_cur.T)
        lidar_cur_tmp += T_cur_to_ref[0:3, 3].reshape((3, 1))

        new_transformation = self.icp_point_to_plane_open3d(lidar_cur_tmp.T, lidar_ref)
        new_transformation = copy.deepcopy(new_transformation)
        T_cur_to_ref = np.matmul(new_transformation, T_cur_to_ref)

        return T_cur_to_ref

    def get_standing_car_candidates(self, transformations):
        car_locations = []
        car_locations_lidar = []
        car_locations_masks = []

        precomputed_masks = self.precompute_detectron_kitti()

        for i in range(-self.cfg.frames_creation.nscans_before, self.cfg.frames_creation.nscans_after + 1):
            # Ignore the reference scan and also do not search for not existing datas

            path_to_cur_velo = self.path_to_folder + 'velodyne_points/data/' + f'{self.file_number + i :0>10}' + '.bin'

            if self.file_number + i < 0 or self.file_number + i >= len(self.kitti_data.oxts) or not os.path.exists(path_to_cur_velo):
                car_locations.append([])
                car_locations_lidar.append([])
                car_locations_masks.append([])
                continue

            T_cur_to_ref = transformations[i + self.cfg.frames_creation.nscans_before, :, :]

            # Load the velo scan
            lidar_cur = np.array(load_velo_scan(path_to_cur_velo))
            lidar_cur = self.prepare_scan(self.file_name, self.img, lidar_cur, save=False)

            masks = precomputed_masks[i + self.cfg.frames_creation.nscans_before]
            car_loc, lidar_points, masks = self.get_car_locations_from_img(lidar_cur, T_cur_to_ref, masks)
            car_locations.append(car_loc)
            car_locations_lidar.append(lidar_points)
            car_locations_masks.append(masks)

        return car_locations, car_locations_lidar, car_locations_masks

    def precompute_detectron_waymo(self):
        if self.generate_raw_masks_or_tracking:
            img_arr = []
            for i in range(0, len(self.waymo_frame)):
                arr_temp = []
                images_sorted = sorted(self.waymo_frame[i].images, key=lambda z: z.name)
                for index, image in enumerate(images_sorted):
                    decoded_image = tf.image.decode_jpeg(image.image).numpy()

                    # Open the image, convert
                    img = np.array(decoded_image, dtype=np.uint8)
                    arr_temp.append(np.moveaxis(img, -1, 0))  # the model expects the image to be in channel first format

                img_arr.append(arr_temp)

            for i in range(len(img_arr)):
                if not self.cfg.general.supress_debug_prints:
                    print("Processing image: ", i, " from ", len(img_arr))
                tmp_img_arr = img_arr[i]
                for z in range(len(tmp_img_arr)):
                    if os.path.exists(self.cfg.paths.merged_frames_path + "masks_raw/" + self.file_name + "/" + str(i) + "_" + str(z) + '.npz'):
                        continue
                    out_dete = self.run_detectron(tmp_img_arr[z], save=False)
                    out_dete = out_dete[out_dete.scores > self.cfg.filtering.score_detectron_thresh]
                    masks_to_save = []
                    for k in range(len(out_dete.pred_masks)):
                        # We are only interested in cars
                        if out_dete.pred_classes[k] == 2 or out_dete.pred_classes[k] == 7:
                            # Take the mask and transpose it
                            mask = np.array(out_dete.pred_masks[k].cpu()).transpose()
                            masks_to_save.append(mask)

                    masks_to_save = np.array(masks_to_save)

                    if not os.path.isdir(self.cfg.paths.merged_frames_path + "masks_raw/" + self.file_name):
                        os.mkdir(self.cfg.paths.merged_frames_path + "masks_raw/" + self.file_name)
                    np.savez_compressed(self.cfg.paths.merged_frames_path + "masks_raw/" + self.file_name + "/" + str(i) + "_" + str(z) + ".npz", np.float32(masks_to_save))

            return None
        else:
            masks_arr = []
            for i in range(0, len(self.waymo_frame)):
                tmp_masks_arr = []
                for z in range(5):
                    masks_raw = np.load(
                        self.cfg.paths.merged_frames_path + "masks_raw/" + self.file_name + "/" + str(i) + "_" + str(z) + '.npz')
                    masks_raw = [np.bool_(masks_raw[key]) for key in masks_raw]
                    tmp_masks_arr.append(masks_raw)
                masks_arr.append(tmp_masks_arr)

            return masks_arr

    def precompute_detectron_kitti(self):
        if self.generate_raw_masks_or_tracking:
            img_arr = []
            #img_arr_visu = []
            for i in range(-self.cfg.frames_creation.nscans_before, self.cfg.frames_creation.nscans_after + 1):
                path_to_cur_img = self.path_to_folder + 'image_02/data/' + f'{self.file_number + i :0>10}' + '.png'
                if self.file_number + i < 0 or self.file_number + i >= len(self.kitti_data.oxts) or not os.path.exists(path_to_cur_img):
                    continue
                img_cv2 = cv2.imread(path_to_cur_img)
                img_cv2 = np.array(img_cv2, dtype=np.uint8)
                #img_arr_visu.append(img_cv2)
                img_cv2 = np.moveaxis(img_cv2, -1, 0)  # the model expects the image to be in channel first format
                img_arr.append(img_cv2)

            detectron_output_arr_tmp = []
            detectron_output_arr = []
            # Get all the outputs from detectron/SAM
            num_of_full_batches = ((len(img_arr)) // self.cfg.general.batch_size)
            for i in range(num_of_full_batches):
                tmp_img_arr = []
                for z in range(self.cfg.general.batch_size):
                    tmp_img_arr.append(img_arr[z + (i * self.cfg.general.batch_size)])
                if self.cfg.frames_creation.use_SAM:
                    out_dete = self.run_SAM_batch(tmp_img_arr)
                else:
                    out_dete = self.run_detectron_batch(tmp_img_arr)
                for z in range(len(out_dete)):
                    detectron_output_arr_tmp.append(out_dete[z])

            last_batch = ((len(img_arr)) % self.cfg.general.batch_size)
            if last_batch > 0:
                tmp_img_arr = []
                for z in range(last_batch):
                    tmp_img_arr.append(img_arr[z + (num_of_full_batches * self.cfg.general.batch_size)])
                if self.cfg.frames_creation.use_SAM:
                    out_dete = self.run_SAM_batch(tmp_img_arr)
                else:
                    out_dete = self.run_detectron_batch(tmp_img_arr)
                for z in range(len(out_dete)):
                    detectron_output_arr_tmp.append(out_dete[z])

            # Merge detector output and missing output
            idx = 0
            for i in range(-self.cfg.frames_creation.nscans_before, self.cfg.frames_creation.nscans_after + 1):
                path_to_cur_img = self.path_to_folder + 'image_02/data/' + f'{self.file_number + i :0>10}' + '.png'
                if self.file_number + i < 0 or self.file_number + i >= len(
                        self.kitti_data.oxts) or not os.path.exists(
                        path_to_cur_img):
                    detectron_output_arr.append([])
                else:
                    detectron_output_arr.append(detectron_output_arr_tmp[idx])
                    idx += 1

            for i in range(len(detectron_output_arr)):
                if len(detectron_output_arr[i]) > 0:
                    detections = detectron_output_arr[i]
                    detections = detections[detections.scores > self.cfg.filtering.score_detectron_thresh]
                    masks_to_save = []
                    for k in range(len(detections.pred_masks)):
                        if detections.pred_classes[k] == 2 or detections.pred_classes[k] == 7:
                            # Take the mask and transpose it
                            mask = np.array(detections.pred_masks[k].cpu()).transpose()
                            masks_to_save.append(mask)

                            #img_arr_visu[i][mask.T] = [255, 0, 0]

                    #result_image = Image.fromarray(img_arr_visu[i])
                    '''
                    # Display the result
                    plt.imshow(result_image)
                    plt.axis('off')  # Hide the axes
                    plt.show()
                    '''
                    masks_to_save = np.array(masks_to_save)
                    detectron_output_arr[i] = masks_to_save

            compressed_arr = zstd.compress(pickle.dumps(detectron_output_arr, pickle.HIGHEST_PROTOCOL))

            with open(self.cfg.paths.merged_frames_path + "masks_raw/" + self.file_name + ".zstd", 'wb') as f:
                f.write(compressed_arr)
            return detectron_output_arr
        else:
            with open(self.cfg.paths.merged_frames_path + "masks_raw/" + self.file_name + ".zstd", 'rb') as f:
                decompressed_data = zstd.decompress(f.read())
            masks = pickle.loads(decompressed_data)
            return masks

    def get_precomputed_detectron_waymo(self, idx_start, idx_end):
        if idx_start < 0:
            idx_start = 0
        if idx_end >= len(self.waymo_frame):
            idx_end = len(self.waymo_frame)

        return self.prec_detectron_output[idx_start: idx_end]

    def precompute_standing_car_candidates_waymo(self):
        car_locations = []
        car_locations_lidar = []
        car_locations_info = []
        car_locations_masks = []
        detectron_output_arr = self.get_precomputed_detectron_waymo(0, len(self.waymo_lidar))

        for i in range(0, len(self.waymo_lidar)):
            # Load the velo scan
            lidar_cur = self.waymo_lidar[i]

            out_det = detectron_output_arr[i]

            car_loc_tmp = None
            car_locations_lidar_tmp = []
            car_locations_info_tmp = []
            car_locations_masks_tmp = []
            for z in range(len(out_det)):
                out_det_tmp = out_det[z]
                if self.cfg.frames_creation.use_growing_for_point_extraction:
                    car_loc, lidar_points, info, masks = self.get_car_locations_from_img_waymo_growing(z + 1, lidar_cur, i, out_det_tmp)
                else:
                    car_loc, lidar_points, info, masks = self.get_car_locations_from_img_waymo(z + 1, lidar_cur, i, out_det_tmp)
                if len(car_loc) > 0:
                    if car_loc_tmp is None:
                        car_loc_tmp = car_loc
                    else:
                        car_loc_tmp = np.concatenate((car_loc_tmp, car_loc), axis=0)
                for k in range(len(lidar_points)):
                    car_locations_lidar_tmp.append(lidar_points[k])
                    car_locations_info_tmp.append(info[k])
                    car_locations_masks_tmp.append(masks[k])
            if car_loc_tmp is None:
                car_loc_tmp = []
            car_locations.append(car_loc_tmp)
            car_locations_lidar.append(car_locations_lidar_tmp)
            car_locations_info.append(car_locations_info_tmp)
            car_locations_masks.append(car_locations_masks_tmp)
        if not self.cfg.general.supress_debug_prints:
            print("New Car Locations")
        self.car_locations = car_locations
        self.car_locations_lidar = car_locations_lidar
        self.car_locations_info = car_locations_info
        self.car_locations_masks = car_locations_masks

        return None

    def convert_to_current_frame(self, transformations):
        car_locations = []
        car_locations_lidar = []
        car_locations_info = []
        car_locations_masks = []
        detectron_output_arr_tmp = self.get_precomputed_detectron_waymo(self.pic_index - self.cfg.frames_creation.nscans_before,
                                                                        self.pic_index + self.cfg.frames_creation.nscans_after + 1)
        detectron_output_arr = []

        # Merge detector output and missing output
        idx = 0
        for i in range(-self.cfg.frames_creation.nscans_before, self.cfg.frames_creation.nscans_after + 1):
            if self.pic_index + i < 0 or self.pic_index + i >= len(self.waymo_frame) or len(self.car_locations[self.pic_index + i]) == 0:
                detectron_output_arr.append([])
            else:
                detectron_output_arr.append(detectron_output_arr_tmp[idx])
                idx += 1

        for i in range(-self.cfg.frames_creation.nscans_before, self.cfg.frames_creation.nscans_after + 1):
            # Ignore the reference scan and also do not search for not existing data

            if self.pic_index + i < 0 or self.pic_index + i >= len(self.waymo_frame) or len(self.car_locations[self.pic_index + i]) == 0:
                car_locations.append([])
                car_locations_lidar.append([])
                car_locations_info.append([])
                car_locations_masks.append([])
                continue

            T_cur_to_ref = transformations[i + self.cfg.frames_creation.nscans_before, :, :]

            tmp_arr = np.matmul(T_cur_to_ref[:3, :3], self.car_locations[self.pic_index + i].T).T
            tmp_arr += T_cur_to_ref[0:3, 3]
            car_locations.append(tmp_arr)

            tmp_arr = []
            for z in range(len(self.car_locations_lidar[self.pic_index + i])):
                tmp_arr.append(np.matmul(T_cur_to_ref[:3, :3], self.car_locations_lidar[self.pic_index + i][z].T).T)
                tmp_arr[z] += T_cur_to_ref[0:3, 3]
            car_locations_lidar.append(tmp_arr)
            car_locations_info.append(self.car_locations_info[self.pic_index + i])
            car_locations_masks.append(self.car_locations_masks[self.pic_index + i])

        return car_locations, car_locations_lidar, car_locations_info, car_locations_masks, detectron_output_arr

    def perform_3D_tracking(self, standing_cars_candidates, car_locations_lidar, car_locations_info, car_locations_masks):
        ref_cars = standing_cars_candidates[0]  # Reference locations of the cars, including the standing cars
        ref_cars_lidar = car_locations_lidar[0]
        ref_cars_info = car_locations_info[0]
        ref_cars_mask = car_locations_masks[0]

        final_moving_cars = []
        final_moving_cars_lidar = []
        final_moving_cars_info = []
        final_moving_cars_masks = []

        if len(ref_cars) > 0:
            tmp_arr = []
            tmp_arr_lidar = []
            tmp_arr_info = []
            tmp_arr_mask = []
            for z in range(len(ref_cars)):
                tmp_arr.append([np.append(ref_cars[z], -self.cfg.frames_creation.nscans_before)])
                tmp_arr_lidar.append([ref_cars_lidar[z]])
                tmp_arr_info.append([ref_cars_info[z]])
                tmp_arr_mask.append([ref_cars_mask[z]])
            moving_cars = tmp_arr
            moving_cars_lidar = tmp_arr_lidar
            moving_cars_info = tmp_arr_info
            moving_cars_masks = tmp_arr_mask

        else:
            moving_cars = []
            moving_cars_lidar = []
            moving_cars_info = []
            moving_cars_masks = []

        #Now lets find moving cars
        for i in range(-self.cfg.frames_creation.nscans_before + 1, self.cfg.frames_creation.nscans_after + 1):
            #Take the cars in the current frame
            cur_cars = standing_cars_candidates[i + self.cfg.frames_creation.nscans_before]
            cur_cars_lidar = car_locations_lidar[i + self.cfg.frames_creation.nscans_before]
            cur_cars_info = car_locations_info[i + self.cfg.frames_creation.nscans_before]
            cur_cars_masks = car_locations_masks[i + self.cfg.frames_creation.nscans_before]
            # Create a mask which looks if the all cars have been found in this frame, otherwise we will discard them and save them
            mask = np.zeros(len(moving_cars), dtype=np.bool_)
            if cur_cars is not None and len(cur_cars) > 0:
                cur_detected_cars = cur_cars
                cur_detected_cars_lidar = cur_cars_lidar
                cur_detected_cars_info = cur_cars_info
                cur_detected_cars_masks = cur_cars_masks

                moving_cars_estimate_locations = []
                for z in range(len(moving_cars)):
                    moving_car = moving_cars[z]
                    # In this case, this is the first time we have seen this car so we cannot predict velocity
                    if len(moving_car) == 1:
                        moving_cars_estimate_locations.append(moving_car[0][:3])
                    else:
                        # Last location + velocity from the last frame
                        est1 = np.array(moving_car[-1][:3] - np.array(moving_car[-2][:3]))
                        if len(moving_car) > 2:
                            est2 = np.array(moving_car[-2][:3] - np.array(moving_car[-3][:3]))
                            if len(moving_car) > 3:
                                est3 = np.array(moving_car[-3][:3] - np.array(moving_car[-4][:3]))
                                if len(moving_car) > 4:
                                    est4 = np.array(moving_car[-4][:3] - np.array(moving_car[-5][:3]))
                                    est = (est1 + est2 + est3 + est4) / 4
                                else:
                                    est = (est1 + est2 + est3) / 3
                            else:
                                est = (est1 + est2) / 2
                        else:
                            est = est1

                        est += np.array(moving_car[-1][:3])

                        moving_cars_estimate_locations.append(est.tolist())
                # Now we got the distance matrix and now we want to do the matching
                new_moving_cars = []
                new_moving_cars_lidar = []
                new_moving_cars_info = []
                new_moving_cars_masks = []

                if len(moving_cars) > 0 and len(cur_detected_cars) > 0:
                    dists = cdist(cur_detected_cars, moving_cars_estimate_locations)
                    mins_cur_to_mov = np.argmin(dists, axis=1)
                    mins_mov_to_cur = np.argmin(dists, axis=0)

                    for z in range(len(cur_detected_cars)):
                        #Check if the closest cur is also closest to the moving car
                        if mins_mov_to_cur[mins_cur_to_mov[z]] == z or True:
                            dist = np.linalg.norm(cur_detected_cars[z][:2] - moving_cars_estimate_locations[mins_cur_to_mov[z]][:2])
                            if dist < self.cfg.frames_creation.dist_treshold_tracking:
                                mask[mins_cur_to_mov[z]] = True
                                moving_cars[mins_cur_to_mov[z]].append(np.append(cur_detected_cars[z], i))
                                moving_cars_lidar[mins_cur_to_mov[z]].append(cur_detected_cars_lidar[z])
                                moving_cars_info[mins_cur_to_mov[z]].append(cur_detected_cars_info[z])
                                moving_cars_masks[mins_cur_to_mov[z]].append(cur_detected_cars_masks[z])
                            else:
                                new_moving_cars.append([np.append(cur_detected_cars[z], i)])
                                new_moving_cars_lidar.append([cur_detected_cars_lidar[z]])
                                new_moving_cars_info.append([cur_detected_cars_info[z]])
                                new_moving_cars_masks.append([cur_detected_cars_masks[z]])
                        # We didnt find a match so it is probably a new moving car
                        else:
                            new_moving_cars.append([np.append(cur_detected_cars[z], i)])
                            new_moving_cars_lidar.append([cur_detected_cars_lidar[z]])
                            new_moving_cars_info.append([cur_detected_cars_info[z]])
                            new_moving_cars_masks.append([cur_detected_cars_masks[z]])
                else:
                    for z in range(len(cur_detected_cars)):
                        new_moving_cars.append([np.append(cur_detected_cars[z], i)])
                        new_moving_cars_lidar.append([cur_detected_cars_lidar[z]])
                        new_moving_cars_info.append([cur_detected_cars_info[z]])
                        new_moving_cars_masks.append([cur_detected_cars_masks[z]])
                index_to_keep = []
                for z in range(len(moving_cars)):
                    if not mask[z] and False:
                        # We did not find a corresponding car so we think we cannot track it anymore
                        final_moving_cars.append(moving_cars[z])
                        final_moving_cars_lidar.append(moving_cars_lidar[z])
                        final_moving_cars_info.append(moving_cars_info[z])
                        final_moving_cars_masks.append(moving_cars_masks[z])
                    else:
                        index_to_keep.append(z)
                tmp_moving_cars = []
                tmp_moving_cars_lidar = []
                tmp_moving_cars_info = []
                tmp_moving_cars_masks = []
                for z in index_to_keep:
                    tmp_moving_cars.append(moving_cars[z])
                    tmp_moving_cars_lidar.append(moving_cars_lidar[z])
                    tmp_moving_cars_info.append(moving_cars_info[z])
                    tmp_moving_cars_masks.append(moving_cars_masks[z])
                moving_cars = tmp_moving_cars
                moving_cars_lidar = tmp_moving_cars_lidar
                moving_cars_info = tmp_moving_cars_info
                moving_cars_masks = tmp_moving_cars_masks

                moving_cars = moving_cars + new_moving_cars
                moving_cars_lidar = moving_cars_lidar + new_moving_cars_lidar
                moving_cars_info = moving_cars_info + new_moving_cars_info
                moving_cars_masks = moving_cars_masks + new_moving_cars_masks

        for z in range(len(moving_cars)):
            final_moving_cars.append(moving_cars[z])
            final_moving_cars_lidar.append(moving_cars_lidar[z])
            final_moving_cars_info.append(moving_cars_info[z])
            final_moving_cars_masks.append(moving_cars_masks[z])

        return final_moving_cars, final_moving_cars_lidar, final_moving_cars_info, final_moving_cars_masks

    def perform_3D_tracking_kitti(self, standing_cars_candidates, car_locations_lidar, car_locations_masks):
        ref_cars = standing_cars_candidates[0]  # Reference locations of the cars, including the standing cars
        ref_cars_lidar = car_locations_lidar[0]
        ref_cars_mask = car_locations_masks[0]

        final_moving_cars = []
        final_moving_cars_lidar = []
        final_moving_cars_masks = []

        if len(ref_cars) > 0:
            tmp_arr = []
            tmp_arr_lidar = []
            tmp_arr_mask = []
            for z in range(len(ref_cars)):
                tmp_arr.append([np.append(ref_cars[z], -self.cfg.frames_creation.nscans_before)])
                tmp_arr_lidar.append([ref_cars_lidar[z]])
                tmp_arr_mask.append([ref_cars_mask[z]])
            moving_cars = tmp_arr
            moving_cars_lidar = tmp_arr_lidar
            moving_cars_masks = tmp_arr_mask

        else:
            moving_cars = []
            moving_cars_lidar = []
            moving_cars_masks = []

        #Now lets find moving cars
        for i in range(-self.cfg.frames_creation.nscans_before + 1, self.cfg.frames_creation.nscans_after + 1):
            #Take the cars in the current frame
            cur_cars = standing_cars_candidates[i + self.cfg.frames_creation.nscans_before]
            cur_cars_lidar = car_locations_lidar[i + self.cfg.frames_creation.nscans_before]
            cur_cars_masks = car_locations_masks[i + self.cfg.frames_creation.nscans_before]
            # Create a mask which looks if the all cars have been found in this frame, otherwise we will discard them and save them
            mask = np.zeros(len(moving_cars), dtype=np.bool_)
            if cur_cars is not None and len(cur_cars) > 0:
                cur_detected_cars = cur_cars
                cur_detected_cars_lidar = cur_cars_lidar
                cur_detected_cars_masks = cur_cars_masks

                moving_cars_estimate_locations = []
                for z in range(len(moving_cars)):
                    moving_car = moving_cars[z]
                    # In this case, this is the first time we have seen this car so we cannot predict velocity
                    if len(moving_car) == 1:
                        moving_cars_estimate_locations.append(moving_car[0][:3])
                    else:
                        # Last location + velocity from the last frame
                        est1 = np.array(moving_car[-1][:3] - np.array(moving_car[-2][:3]))
                        if len(moving_car) > 2:
                            est2 = np.array(moving_car[-2][:3] - np.array(moving_car[-3][:3]))
                            if len(moving_car) > 3:
                                est3 = np.array(moving_car[-3][:3] - np.array(moving_car[-4][:3]))
                                if len(moving_car) > 4:
                                    est4 = np.array(moving_car[-4][:3] - np.array(moving_car[-5][:3]))
                                    est = (est1 + est2 + est3 + est4) / 4
                                else:
                                    est = (est1 + est2 + est3) / 3
                            else:
                                est = (est1 + est2) / 2
                        else:
                            est = est1

                        est += np.array(moving_car[-1][:3])

                        moving_cars_estimate_locations.append(est.tolist())
                # Now we got the distance matrix and now we want to do the matching
                new_moving_cars = []
                new_moving_cars_lidar = []
                new_moving_cars_masks = []

                if len(moving_cars) > 0 and len(cur_detected_cars) > 0:
                    dists = cdist(cur_detected_cars, moving_cars_estimate_locations)
                    mins_cur_to_mov = np.argmin(dists, axis=1)
                    mins_mov_to_cur = np.argmin(dists, axis=0)

                    for z in range(len(cur_detected_cars)):
                        #Check if the closest cur is also closest to the moving car
                        if mins_mov_to_cur[mins_cur_to_mov[z]] == z:
                            dist = np.linalg.norm(cur_detected_cars[z][:3] - moving_cars_estimate_locations[mins_cur_to_mov[z]][:3])
                            if dist < self.cfg.frames_creation.dist_treshold_tracking:
                                mask[mins_cur_to_mov[z]] = True
                                moving_cars[mins_cur_to_mov[z]].append(np.append(cur_detected_cars[z], i))
                                moving_cars_lidar[mins_cur_to_mov[z]].append(cur_detected_cars_lidar[z])
                                moving_cars_masks[mins_cur_to_mov[z]].append(cur_detected_cars_masks[z])
                            else:
                                new_moving_cars.append([np.append(cur_detected_cars[z], i)])
                                new_moving_cars_lidar.append([cur_detected_cars_lidar[z]])
                                new_moving_cars_masks.append([cur_detected_cars_masks[z]])
                        # We didnt find a match so it is probably a new moving car
                        else:
                            new_moving_cars.append([np.append(cur_detected_cars[z], i)])
                            new_moving_cars_lidar.append([cur_detected_cars_lidar[z]])
                            new_moving_cars_masks.append([cur_detected_cars_masks[z]])
                else:
                    for z in range(len(cur_detected_cars)):
                        new_moving_cars.append([np.append(cur_detected_cars[z], i)])
                        new_moving_cars_lidar.append([cur_detected_cars_lidar[z]])
                        new_moving_cars_masks.append([cur_detected_cars_masks[z]])
                index_to_keep = []
                for z in range(len(moving_cars)):
                    if not mask[z] and False:
                        # We did not find a corresponding car so we think we cannot track it anymore
                        final_moving_cars.append(moving_cars[z])
                        final_moving_cars_lidar.append(moving_cars_lidar[z])
                        final_moving_cars_masks.append(moving_cars_masks[z])
                    else:
                        index_to_keep.append(z)
                tmp_moving_cars = []
                tmp_moving_cars_lidar = []
                tmp_moving_cars_masks = []
                for z in index_to_keep:
                    tmp_moving_cars.append(moving_cars[z])
                    tmp_moving_cars_lidar.append(moving_cars_lidar[z])
                    tmp_moving_cars_masks.append(moving_cars_masks[z])
                moving_cars = tmp_moving_cars
                moving_cars_lidar = tmp_moving_cars_lidar
                moving_cars_masks = tmp_moving_cars_masks

                moving_cars = moving_cars + new_moving_cars
                moving_cars_lidar = moving_cars_lidar + new_moving_cars_lidar
                moving_cars_masks = moving_cars_masks + new_moving_cars_masks

        for z in range(len(moving_cars)):
            final_moving_cars.append(moving_cars[z])
            final_moving_cars_lidar.append(moving_cars_lidar[z])
            final_moving_cars_masks.append(moving_cars_masks[z])

        return final_moving_cars, final_moving_cars_lidar, final_moving_cars_masks

    def decide_if_standing_or_moving(self, cars, waymo=True):
        for i in range(len(cars)):
            start = None
            for z in range(len(cars[i].locations)):
                if cars[i].locations[z] is not None:
                    start = cars[i].locations[z]
                    break

            end = None
            for z in range(len(cars[i].locations)):
                if cars[i].locations[-(z + 1)] is not None:
                    end = cars[i].locations[-(z + 1)]
                    break

            if start is None or end is None:
                cars[i].moving = False
            else:
                if waymo:
                    dist_traveled = (np.power(start[0] - end[0], 2) +
                                     np.power(start[1] - end[1], 2))
                else:
                    dist_traveled = (np.power(start[0] - end[0], 2) +
                                     np.power(start[2] - end[2], 2))

                if np.sqrt(dist_traveled) > self.cfg.frames_creation.dist_treshold_moving:
                    cars[i].moving = True
                else:
                    cars[i].moving = False

            if waymo:
                for z in range(len(cars[i].locations)):
                    if cars[i].locations[z] is not None:
                        cars[i].locations[z] = cars[i].locations[z][:3]

        return cars

    def load_merged_frames_from_files_waymo_tracker(self, track2D=False, merge_two_trackers=False):
        self.lidar = self.waymo_lidar[self.pic_index].T

        if merge_two_trackers:
            if self.cfg.frames_creation.use_growing_for_point_extraction:
                with open(self.cfg.paths.merged_frames_path + "cars_2DTrack_growing/" + self.file_name + "/" + str(self.pic_index) + ".zstd", 'rb') as f:
                    decompressed_data = zstd.decompress(f.read())
                cars2D = pickle.loads(decompressed_data)
                with open(self.cfg.paths.merged_frames_path + "cars_3DTrack_growing/" + self.file_name + "/" + str(self.pic_index) + ".zstd", 'rb') as f:
                    decompressed_data = zstd.decompress(f.read())
                cars3D = pickle.loads(decompressed_data)
            else:
                with open(self.cfg.paths.merged_frames_path + "cars_2DTrack/" + self.file_name + "/" + str(self.pic_index) + ".zstd", 'rb') as f:
                    decompressed_data = zstd.decompress(f.read())
                cars2D = pickle.loads(decompressed_data)
                with open(self.cfg.paths.merged_frames_path + "cars_3DTrack/" + self.file_name + "/" + str(self.pic_index) + ".zstd", 'rb') as f:
                    decompressed_data = zstd.decompress(f.read())
                cars3D = pickle.loads(decompressed_data)
            self.cars = cars2D + cars3D
            self.cars = sorted(self.cars, key=lambda x: x.lidar.shape[1], reverse=True)
            self.cars3D_start = len(cars2D)

            #Make the optimization area significantly smaller, as we dont need so much precision for the merging :)
            self.opt_param1_iters = 20  # X
            self.opt_param2_iters = 20  # Y or Z, depending on dataset
            self.opt_param3_iters = 20  # Theta

        elif track2D:
            if self.cfg.frames_creation.use_growing_for_point_extraction:
                with open(self.cfg.paths.merged_frames_path + "cars_2DTrack_growing/" + self.file_name + "/" + str(self.pic_index) + ".zstd", 'rb') as f:
                    decompressed_data = zstd.decompress(f.read())
            else:
                with open(self.cfg.paths.merged_frames_path + "cars_2DTrack/" + self.file_name + "/" + str(self.pic_index) + ".zstd", 'rb') as f:
                    decompressed_data = zstd.decompress(f.read())
            cars = pickle.loads(decompressed_data)
            self.cars = cars
            self.cars = sorted(self.cars, key=lambda x: x.lidar.shape[0], reverse=True)
        else:
            if self.cfg.frames_creation.use_growing_for_point_extraction:
                with open(self.cfg.paths.merged_frames_path + "cars_3DTrack_growing/" + self.file_name + "/" + str(self.pic_index) + ".zstd", 'rb') as f:
                    decompressed_data = zstd.decompress(f.read())
            else:
                with open(self.cfg.paths.merged_frames_path + "cars_3DTrack/" + self.file_name + "/" + str(self.pic_index) + ".zstd", 'rb') as f:
                    decompressed_data = zstd.decompress(f.read())
            cars = pickle.loads(decompressed_data)
            self.cars = cars
            self.cars = sorted(self.cars, key=lambda x: x.lidar.shape[0], reverse=True)

        cur_lidar = np.ascontiguousarray(self.lidar.T[:,:3]).astype('float32')
        quantizer = faiss.IndexFlatL2(cur_lidar.shape[1])
        index_faiss = faiss.IndexIVFFlat(quantizer, cur_lidar.shape[1], int(np.floor(np.sqrt(cur_lidar.shape[0]))))
        index_faiss.train(cur_lidar)
        index_faiss.add(cur_lidar)
        index_faiss.nprobe = 10
        
        new_cars = []

        for car in self.cars:
            if car.lidar is not None:
                center = np.zeros((1, 3))
                center[0, 0] = np.median(car.lidar[:, 0])
                center[0, 1] = np.median(car.lidar[:, 1])
                center[0, 2] = np.median(car.lidar[:, 2])

                idx, distances, indexes = index_faiss.range_search(np.ascontiguousarray(center).astype('float32'), 2. ** 2)

                if len(distances) < 1:
                    continue

                padding = np.ones((car.lidar.shape[0], 3))

                car.lidar = np.concatenate((car.lidar, padding), axis=1).T
                new_cars.append(car)
        self.cars = new_cars

        '''
        for car in self.cars:
            if car.lidar is not None:
                padding = np.ones((car.lidar.shape[0], 3))
                car.lidar = np.concatenate((car.lidar, padding), axis=1).T
        '''

    def non_maxima_surpression(self, cars):
        num_of_cars = len(cars)

        indx = 0
        to_be_optimized = []

        while indx < num_of_cars:
            num_of_cars = len(cars)
            if cars[indx].lidar is None or cars[indx].optimized is False:
                indx += 1
                continue
            else:
                bbox_ref_center = [cars[indx].x, cars[indx].y, cars[indx].z]
                if self.args.dataset == 'kitti':
                    bbox_ref_size = [cars[indx].width, cars[indx].height, cars[indx].length]
                elif self.args.dataset == 'waymo':
                    bbox_ref_size = [cars[indx].length, cars[indx].width, cars[indx].height]
                else:
                    raise ValueError('Dataset not supported')
                bbox_ref_theta = cars[indx].theta

                scaled_cube = np.diag(bbox_ref_size).dot(self.unit_cube.T).T
                if self.args.dataset == 'kitti':
                    rotation = R.from_euler('y', bbox_ref_theta, degrees=False)
                elif self.args.dataset == 'waymo':
                    rotation = R.from_euler('z', bbox_ref_theta, degrees=False)
                else:
                    raise ValueError('Dataset not supported')

                rotated_cube = rotation.apply(scaled_cube)
                bbox_ref_points = rotated_cube + bbox_ref_center

                for i in range(indx + 1, num_of_cars):
                    if cars[i].lidar is None or not cars[i].optimized:
                        continue
                    else:
                        bbox_cur_center = [cars[i].x, cars[i].y, cars[i].z]
                        if self.args.dataset == 'kitti':
                            bbox_cur_size = [cars[i].width, cars[i].height, cars[i].length]
                        elif self.args.dataset == 'waymo':
                            bbox_cur_size = [cars[i].length, cars[i].width, cars[i].height]
                        else:
                            raise ValueError('Dataset not supported')
                        bbox_cur_theta = cars[i].theta

                        scaled_cube = np.diag(bbox_cur_size).dot(self.unit_cube.T).T
                        if self.args.dataset == 'kitti':
                            rotation = R.from_euler('y', bbox_cur_theta, degrees=False)
                        elif self.args.dataset == 'waymo':
                            rotation = R.from_euler('z', bbox_cur_theta, degrees=False)
                        else:
                            raise ValueError('Dataset not supported')
                        rotated_cube = rotation.apply(scaled_cube)
                        bbox_cur_points = rotated_cube + bbox_cur_center

                        vol, iou = pytorch3d.ops.box3d_overlap(
                            torch.tensor(bbox_ref_points, dtype=torch.float32).unsqueeze(0),
                            torch.tensor(bbox_cur_points, dtype=torch.float32).unsqueeze(0))
                        if iou[0].item() > self.cfg.optimization.nms_threshold:
                            if self.cfg.optimization.nms_merge_and_reopt:
                                cars[indx].lidar = np.concatenate((cars[indx].lidar, cars[i].lidar), axis=1)
                                to_be_optimized.append(indx)
                            cars[i].lidar = None
                            cars[i].optimized = False
                indx += 1

        if self.cfg.optimization.merge_two_trackers:
            to_be_optimized = np.array(list(range(len(cars))))
            self.opt_param1_iters = 40  # X
            self.opt_param2_iters = 40  # Y or Z, depending on dataset
            self.opt_param3_iters = 40  # Theta
        else:
            to_be_optimized = np.array(to_be_optimized)
        unique = np.unique(to_be_optimized)

        return cars, unique

    # Function which returns the location of the cars in the picture. The locations are a median of points detected
    def get_car_locations_from_img(self, scan, T_cur_to_ref, masks):
        transformed_means = []
        lidar_points = []
        out_masks = []

        for z in range(len(masks)):
            mask = masks[z]
            mask_old = copy.deepcopy(mask)
            # Shrink the mask to approx half of the area to avoid detecting outliers as standing cars
            struct_size = int(2 + np.sqrt(np.count_nonzero(mask)) // 10)

            mask = np.invert(mask)
            mask = scipy.ndimage.binary_dilation(mask, iterations=struct_size)
            mask = np.invert(mask)

            # Now, get indexes of the points which project into the mask
            tmp1 = np.argwhere(mask[scan[4, :].astype(int), scan[5, :].astype(int)])

            # Now, filter the points based on the indexes
            filtered_lidar = np.array([scan[0, tmp1], scan[1, tmp1], scan[2, tmp1]]).transpose()[0]

            # Sometimes, we just lack the number of points, so if it is small, just skip it
            if filtered_lidar.shape[0] < self.cfg.filtering.moving_detection_threshold:
                struct_size = 1
                mask = np.invert(copy.deepcopy(mask_old))
                mask = scipy.ndimage.binary_dilation(mask, iterations=struct_size)
                mask = np.invert(mask)
                # Now, get indexes of the points which project into the mask
                tmp1 = np.argwhere(mask[scan[4, :].astype(int), scan[5, :].astype(int)])

                # Now, filter the points based on the indexes
                filtered_lidar = np.array([scan[0, tmp1], scan[1, tmp1], scan[2, tmp1]]).transpose()[
                    0]

                if filtered_lidar.shape[0] < self.cfg.filtering.moving_detection_threshold:
                    # Now, get indexes of the points which project into the mask
                    tmp1 = np.argwhere(mask_old[scan[4, :].astype(int), scan[5, :].astype(int)])

                    # Now, filter the points based on the indexes
                    filtered_lidar = np.array([scan[0, tmp1], scan[1, tmp1], scan[2, tmp1]]).transpose()[
                        0]

                    if filtered_lidar.shape[0] < self.cfg.filtering.moving_detection_threshold:
                        continue

            x_mean, y_mean, z_mean = self.compute_mean(filtered_lidar)

            # Filter by circle
            dist_from_mean = np.sqrt(
                (x_mean - filtered_lidar[:, 0]) ** 2 + (z_mean - filtered_lidar[:, 2]) ** 2)

            indexes = np.argwhere(dist_from_mean < self.cfg.filtering.filter_diameter)

            filtered_lidar = \
                np.array(
                    [filtered_lidar[indexes, 0], filtered_lidar[indexes, 1], filtered_lidar[indexes, 2]]).T[0]

            # look for the mean on the filtered data by circle, which will hopefully get better results
            if filtered_lidar.shape[0] > 0:
                x_mean, y_mean, z_mean = self.compute_mean(filtered_lidar)

            # Transform the points between frames.
            mean_transformed = np.matmul(T_cur_to_ref[0:3, 0:3], np.array([x_mean, y_mean, z_mean]).T).transpose()
            mean_transformed += T_cur_to_ref[0:3, 3]
            # Check if the car is atleast infront of us
            if mean_transformed[2] > 0.:

                # Lets save the lidar points for the moving cars detection.
                # Now, get indexes of the points which project into the mask
                tmp1 = np.argwhere(mask_old[scan[4, :].astype(int), scan[5, :].astype(int)])

                # Now, filter the points based on the indexes
                filtered_lidar = np.array([scan[0, tmp1], scan[1, tmp1], scan[2, tmp1]]).transpose()[
                    0]

                x_mean, y_mean, z_mean = self.compute_mean(filtered_lidar)

                # Filter by circle
                dist_from_mean = np.sqrt(
                    (x_mean - filtered_lidar[:, 0]) ** 2 + (z_mean - filtered_lidar[:, 2]) ** 2)

                indexes = np.argwhere(dist_from_mean < self.cfg.filtering.filter_diameter)

                filtered_lidar = \
                    np.array(
                        [filtered_lidar[indexes, 0], filtered_lidar[indexes, 1], filtered_lidar[indexes, 2]]).T[0]

                # Transform the points between frames.
                filtered_lidar = np.matmul(T_cur_to_ref[0:3, 0:3], filtered_lidar.T).T
                filtered_lidar += T_cur_to_ref[0:3, 3]

                if filtered_lidar.shape[0] < self.cfg.filtering.moving_detection_threshold:
                    continue

                transformed_means.append(mean_transformed)
                lidar_points.append(filtered_lidar)
                out_masks.append(mask_old)
        '''
        if len(lidar_points) > 0:
            # visualize points and lidar
            pcd = open3d.geometry.PointCloud()
            pcd.points = open3d.utility.Vector3dVector(scan.T[:, :3])
            pcd.colors = open3d.utility.Vector3dVector(
                np.zeros((scan.T[:, :3].shape[0], 3)))  # Black color for all points

            filtered_pcd = open3d.geometry.PointCloud()
            filtered_pcd.points = open3d.utility.Vector3dVector(lidar_points[0])
            filtered_pcd.colors = open3d.utility.Vector3dVector(
                np.ones((lidar_points[0].shape[0], 3)) * [1, 0, 0])  # Red color for filtered points

            # Visualize both point clouds together
            open3d.visualization.draw_geometries([pcd, filtered_pcd])
        '''
        return np.array(transformed_means), lidar_points, np.array(out_masks)

    def get_car_locations_from_img_waymo(self, img_index, scan, frame_index, out_det):
        transformed_means = []
        lidar_points = []
        info = []
        masks = []

        scan = scan[scan[:, 3] == img_index]
        scan = scan.T

        out_det = out_det[0]

        for z in range(len(out_det)):
            # Take the mask and transpose it
            mask = np.copy(out_det[z])

            mask_old = copy.deepcopy(mask)
            # Shrink the mask to approx half of the area to avoid detecting outliers as standing cars
            #print("old: ", np.count_nonzero(mask))
            struct_size = int(2 + np.sqrt(np.count_nonzero(mask)) // 10)
            mask = np.invert(mask)
            mask = scipy.ndimage.binary_dilation(mask, iterations=struct_size)
            mask = np.invert(mask)
            # print("new: ", np.count_nonzero(mask))

            # Now, get indexes of the points which project into the mask
            tmp1 = np.argwhere(mask[scan[4, :].astype(int), scan[5, :].astype(int)])

            # Now, filter the points based on the indexes
            filtered_lidar = np.array([scan[0, tmp1], scan[1, tmp1], scan[2, tmp1]]).transpose()[
                0]

            # Sometimes, we just lack the number of points, so if it is small, just skip it
            if filtered_lidar.shape[0] < self.cfg.filtering.moving_detection_threshold:
                struct_size = 1
                mask = np.invert(copy.deepcopy(mask_old))
                mask = scipy.ndimage.binary_dilation(mask, iterations=struct_size)
                mask = np.invert(mask)
                # Now, get indexes of the points which project into the mask
                tmp1 = np.argwhere(mask[scan[4, :].astype(int), scan[5, :].astype(int)])

                # Now, filter the points based on the indexes
                filtered_lidar = np.array([scan[0, tmp1], scan[1, tmp1], scan[2, tmp1]]).transpose()[
                    0]

                if filtered_lidar.shape[0] < self.cfg.filtering.moving_detection_threshold:
                    # Now, get indexes of the points which project into the mask
                    tmp1 = np.argwhere(mask_old[scan[4, :].astype(int), scan[5, :].astype(int)])

                    # Now, filter the points based on the indexes
                    filtered_lidar = np.array([scan[0, tmp1], scan[1, tmp1], scan[2, tmp1]]).transpose()[
                        0]

                    if filtered_lidar.shape[0] < self.cfg.filtering.moving_detection_threshold:
                        continue

            x_mean, y_mean, z_mean = self.compute_mean(filtered_lidar)

            # Filter by circle
            dist_from_mean = np.sqrt(
                (x_mean - filtered_lidar[:, 0]) ** 2 + (y_mean - filtered_lidar[:, 1]) ** 2)

            indexes = np.argwhere(dist_from_mean < self.cfg.filtering.filter_diameter)

            filtered_lidar = \
                np.array(
                    [filtered_lidar[indexes, 0], filtered_lidar[indexes, 1], filtered_lidar[indexes, 2]]).T[0]

            # look for the mean on the filtered data by circle, which will hopefully get better results
            if filtered_lidar.shape[0] > 0:
                x_mean, y_mean, z_mean = self.compute_mean(filtered_lidar)

            # Transform the points between frames.
            #mean_transformed = np.matmul(T_cur_to_ref[0:3, 0:3], np.array([x_mean, y_mean, z_mean]).T).transpose()
            #mean_transformed += T_cur_to_ref[0:3, 3]
            mean_transformed = np.array([x_mean, y_mean, z_mean])

            # Lets save the lidar points for the moving cars detection.
            # Now, get indexes of the points which project into the mask
            tmp1 = np.argwhere(mask_old[scan[4, :].astype(int), scan[5, :].astype(int)])

            # Now, filter the points based on the indexes
            filtered_lidar = np.array([scan[0, tmp1], scan[1, tmp1], scan[2, tmp1]]).transpose()[
                0]

            x_mean, y_mean, z_mean = self.compute_mean(filtered_lidar)

            # Filter by circle
            dist_from_mean = np.sqrt(
                (x_mean - filtered_lidar[:, 0]) ** 2 + (y_mean - filtered_lidar[:, 1]) ** 2)

            indexes = np.argwhere(dist_from_mean < self.cfg.filtering.filter_diameter)

            filtered_lidar = \
                np.array(
                    [filtered_lidar[indexes, 0], filtered_lidar[indexes, 1], filtered_lidar[indexes, 2]]).T[0]

            # Transform the points between frames.
            #filtered_lidar = np.matmul(T_cur_to_ref[0:3, 0:3], filtered_lidar.T).T
            #filtered_lidar += T_cur_to_ref[0:3, 3]

            if filtered_lidar.shape[0] < self.cfg.filtering.moving_detection_threshold:
                continue

            transformed_means.append(mean_transformed)
            lidar_points.append(filtered_lidar)

            info_item = np.array([0, frame_index, img_index, 0])
            info.append(info_item)
            masks.append(out_det[z])

        return np.array(transformed_means), lidar_points, info, np.array(masks)

    def get_car_locations_from_img_waymo_growing(self, img_index, scan, frame_index, out_det):
        transformed_means = []
        lidar_points = []
        info = []
        masks = []

        scan_orig = np.copy(scan)
        out_det = out_det[0]

        for z in range(len(out_det)):
            # Take the mask and transpose it
            mask = np.copy(out_det[z])

            start = time.time_ns()
            filtered_lidar = self.perform_growing(mask, img_index, scan_orig)
            if not self.cfg.general.supress_debug_prints:
                print("Time to perform growing: ", (time.time_ns() - start) / 1000000)

            if filtered_lidar is not None:
                if filtered_lidar.shape[0] < self.cfg.filtering.moving_detection_threshold:
                    continue
                else:
                    x_mean, y_mean, z_mean = self.compute_mean(filtered_lidar)

                    mean_transformed = np.array([x_mean, y_mean, z_mean])

                    transformed_means.append(mean_transformed)
                    lidar_points.append(filtered_lidar)

                    info_item = np.array([0, frame_index, img_index, 0])
                    info.append(info_item)
                    masks.append(out_det[z])

            else:
                continue

        return np.array(transformed_means), lidar_points, info, np.array(masks)

    def prepare_scan(self, filename, img, lidar, save=True, crop=True):
        lidar = self.transform_velo_to_cam(filename, lidar)
        return self.project_lidar_points(lidar, img, save, crop)

    def transform_velo_to_cam(self, filename, lidar, filter_points=True):
        # Now we need homogenous coordinates and we do not care about the reflections
        lidar[:, 3] = 1
        # Transform to the camera coordinate
        lidar = lidar.transpose()
        # This should be rectified already
        T_velo_to_cam = self.kitti_data.calib.T_cam2_velo
        lidar = np.matmul(T_velo_to_cam, lidar)

        if filter_points:
            # Delete all points which are not in front of the camera
            mask = lidar[2, :] > 0.
            lidar = lidar[:, mask]

        self.velo_to_cam = T_velo_to_cam
        self.P2_rect = self.kitti_data.calib.P_rect_00
        return lidar

    def project_lidar_points(self, lidar, img, save=True, crop=True):
        proj_lidar = np.matmul(self.P2_rect, lidar)
        proj_lidar = proj_lidar[0:2, :] / proj_lidar[2, :]

        # Add projected data to the lidar data
        lidar = np.concatenate((lidar, proj_lidar), axis=0)
        lidar[4:6, :] = np.rint(lidar[4:6, :])

        # Filter lidar data based on their projection to the camera: if they actually fit?
        if crop:
            mask_xmin = lidar[4, :] >= 0.
            lidar = lidar[:, mask_xmin]
            mask_xmax = lidar[4, :] < img.shape[2]  # img width
            lidar = lidar[:, mask_xmax]
            mask_ymin = lidar[5, :] >= 0.
            lidar = lidar[:, mask_ymin]
            mask_ymax = lidar[5, :] < img.shape[1]  # img height
            lidar = lidar[:, mask_ymax]

        if save:
            self.lidar = lidar
        else:
            return lidar

    def prepare_img_dist(self, img):
        img_dist = -np.ones((img.shape[2], img.shape[1]))

        for i in range(self.lidar.shape[1]):
            img_dist[self.lidar[4, i].astype(int), self.lidar[5, i].astype(int)] = np.maximum(
                img_dist[self.lidar[4, i].astype(int), self.lidar[5, i].astype(int)],
                self.lidar[0, i] ** 2 + self.lidar[2, i] ** 2)

        # We need to add dilatation, because the lidar points are too sparse on camera
        footprint = np.ones((5, 5))
        img_dist = scipy.ndimage.grey_dilation(img_dist, footprint=footprint)
        self.img_dist = img_dist

    def load_current_segment(self):
        file_name = self.cfg.paths.waymo_path + self.random_indexes[self.segment_index]
        self.file_name = self.random_indexes[self.segment_index]
        dataset = tf.data.TFRecordDataset(file_name, compression_type='')
        if not self.cfg.general.supress_debug_prints:
            print("Segment: ", file_name)

        self.waymo_data = []
        self.waymo_frame = []
        self.waymo_lidar = []

        for i, data in enumerate(dataset):
            if i >10: break
            self.waymo_data.append(data)
            frame = open_dataset.Frame()
            frame.ParseFromString(bytearray(data.numpy()))
            self.waymo_frame.append(frame)

            if self.generate_raw_lidar:
                self.generate_raw_lidar_frame(frame, i)
            else:
                lidar_raw = np.load(
                    self.cfg.paths.merged_frames_path + "lidar_raw/" + self.file_name + "/" + str(i) + '.npz')
                lidar_raw = [lidar_raw[key] for key in lidar_raw][0]

                self.waymo_lidar.append(lidar_raw)

        if not self.cfg.general.supress_debug_prints:
            print("Segment loaded")

    def generate_raw_lidar_frame(self, frame, i):
        (range_images, camera_projections, _,
         range_image_top_pose) = frame_utils.parse_range_image_and_camera_projection(frame)

        points, cp_points = frame_utils.convert_range_image_to_point_cloud(frame, range_images,
                                                                           camera_projections,
                                                                           range_image_top_pose)

        points_all = np.concatenate(points, axis=0)
        cp_points_all = np.concatenate(cp_points, axis=0)

        cp_points_all_concat = np.concatenate([points_all, cp_points_all[..., 0:3]], axis=-1)

        self.waymo_lidar.append(cp_points_all_concat)

        if not os.path.isdir(self.cfg.paths.merged_frames_path + "lidar_raw/" + self.file_name):
            os.mkdir(self.cfg.paths.merged_frames_path + "lidar_raw/" + self.file_name)
        np.savez_compressed(
            self.cfg.paths.merged_frames_path + "lidar_raw/" + self.file_name + "/" + str(i) + ".npz",
            np.float32(cp_points_all_concat))

    def load_lidar_templatesv2(self):
        pcd1, _ = self.load_and_sample_fiat()
        #pcd1, _ = self.load_and_sample_cube()
        pcd2, _ = self.load_and_sample_passat()
        pcd3, _ = self.load_and_sample_suv()
        pcd4, _ = self.load_and_sample_mpv()
        #pcd1_scale, _ = self.load_and_sample_fiat_scale()
        #pcd2_scale, _ = self.load_and_sample_passat_scale()
        #pcd1_scale, _ = self.load_and_sample_cube()
        #pcd2_scale, _ = self.load_and_sample_cube_scale()

        #coord_frame = open3d.geometry.TriangleMesh.create_coordinate_frame(size=1.)  # Only for visu purpose
        #open3d.visualization.draw_geometries([pcd4, coord_frame])

        pcd1 = np.asarray(pcd1.points)
        pcd2 = np.asarray(pcd2.points)
        pcd3 = np.asarray(pcd3.points)
        pcd4 = np.asarray(pcd4.points)
        #pcd1_scale = np.asarray(pcd1_scale.points)
        #pcd2_scale = np.asarray(pcd2_scale.points)

        if self.args.dataset == 'waymo':
            pcd1[:, 2] += self.cfg.templates.offset_fiat
            pcd2[:, 2] += self.cfg.templates.offset_passat
            pcd3[:, 2] += self.cfg.templates.offset_suv
            pcd4[:, 2] += self.cfg.templates.offset_mpv
        else:
            pcd1[:, 1] -= self.cfg.templates.offset_fiat
            pcd2[:, 1] -= self.cfg.templates.offset_passat
            pcd3[:, 1] -= self.cfg.templates.offset_suv
            pcd4[:, 1] -= self.cfg.templates.offset_mpv


        self.lidar_car_template_non_filt = [pcd1, pcd2, pcd3, pcd4]
        self.lidar_car_template_scale = [pcd1, pcd2, pcd3, pcd4]

    def load_and_sample_fiat(self):
        mesh = open3d.io.read_triangle_mesh("../data/fiat2.gltf")  # Read mesh of fiat uno converted via blender
        bbox = mesh.get_minimal_oriented_bounding_box()
        T = np.eye(4)
        T[:3, 3] = (-bbox.center[0], -bbox.center[1], -bbox.center[2])
        mesh.transform(T)

        if self.args.dataset == 'waymo':
            T = np.eye(4)
            T[:3, :3] = open3d.geometry.get_rotation_matrix_from_zxy((np.pi/2, np.pi,0))  # Y rotation -||-
            mesh.transform(T)
        else:
            T = np.eye(4)
            T[:3, :3] = open3d.geometry.get_rotation_matrix_from_zxy((np.pi, np.pi/2, 0))  # Y rotation -||-
            mesh.transform(T)

        vertices = np.asarray(mesh.vertices)
        ax0_size = np.amax(vertices[:, 0]) - np.amin(vertices[:, 0])
        ax1_size = np.amax(vertices[:, 1]) - np.amin(vertices[:, 1])
        ax2_size = np.amax(vertices[:, 2]) - np.amin(vertices[:, 2])

        if self.args.dataset == 'waymo':
            scale_ax0 = self.cfg.templates.template_length / ax0_size
            scale_ax1 = self.cfg.templates.template_width / ax1_size
            scale_ax2 = self.cfg.templates.template_height / ax2_size
        else:
            scale_ax0 = self.cfg.templates.template_width / ax0_size
            scale_ax1 = self.cfg.templates.template_height / ax1_size
            scale_ax2 = self.cfg.templates.template_length / ax2_size

        vertices[:, 0] *= scale_ax0
        vertices[:, 1] *= scale_ax1
        vertices[:, 2] *= scale_ax2

        mesh.vertices = open3d.utility.Vector3dVector(vertices)

        mesh.compute_vertex_normals()  # Looks better in visu with normals
        pcd = mesh.sample_points_uniformly(number_of_points=1000)

        return pcd, mesh

    def load_and_sample_fiat_scale(self):
        mesh = open3d.io.read_triangle_mesh("../data/fiat_scale.gltf")  # Read mesh of fiat uno converted via blender
        bbox = mesh.get_minimal_oriented_bounding_box()
        T = np.eye(4)
        T[:3, 3] = (-bbox.center[0], -bbox.center[1], -bbox.center[2])
        mesh.transform(T)

        if self.args.dataset == 'waymo':
            T = np.eye(4)
            T[:3, :3] = open3d.geometry.get_rotation_matrix_from_zxy((np.pi / 2, 0, -np.pi / 2))  # Y rotation -||-
            mesh.transform(T)
        else:
            T = np.eye(4)
            T[:3, :3] = open3d.geometry.get_rotation_matrix_from_zxy((-np.pi / 2, -np.pi / 2, 0))  # Y rotation -||-
            mesh.transform(T)

        vertices = np.asarray(mesh.vertices)
        ax0_size = np.amax(vertices[:, 0]) - np.amin(vertices[:, 0])
        ax1_size = np.amax(vertices[:, 1]) - np.amin(vertices[:, 1])
        ax2_size = np.amax(vertices[:, 2]) - np.amin(vertices[:, 2])

        if self.args.dataset == 'waymo':
            scale_ax0 = self.cfg.templates.template_length / ax0_size
            scale_ax1 = self.cfg.templates.template_width / ax1_size
            scale_ax2 = self.cfg.templates.template_height / ax2_size
        else:
            scale_ax0 = self.cfg.templates.template_width / ax0_size
            scale_ax1 = self.cfg.templates.template_height / ax1_size
            scale_ax2 = self.cfg.templates.template_length / ax2_size

        vertices[:, 0] *= scale_ax0
        vertices[:, 1] *= scale_ax1
        vertices[:, 2] *= scale_ax2

        mesh.vertices = open3d.utility.Vector3dVector(vertices)

        mesh.compute_vertex_normals()  # Looks better in visu with normals
        pcd = mesh.sample_points_uniformly(number_of_points=1000)

        return pcd, mesh

    def load_and_sample_passat(self):
        mesh = open3d.io.read_triangle_mesh("../data/passat2.gltf")  # Read mesh of fiat uno converted via blender
        bbox = mesh.get_minimal_oriented_bounding_box()
        T = np.eye(4)
        T[:3, 3] = (-bbox.center[0], -bbox.center[1], -bbox.center[2])
        mesh.transform(T)

        if self.args.dataset == 'waymo':
            T = np.eye(4)
            T[:3, :3] = open3d.geometry.get_rotation_matrix_from_zxy((-np.pi/2, np.pi/2, 0))  # Y rotation -||-
            mesh.transform(T)
        else:
            T = np.eye(4)
            T[:3, :3] = open3d.geometry.get_rotation_matrix_from_zxy((np.pi, 0, np.pi))  # Y rotation -||-
            mesh.transform(T)

        vertices = np.asarray(mesh.vertices)
        ax0_size = np.amax(vertices[:, 0]) - np.amin(vertices[:, 0])
        ax1_size = np.amax(vertices[:, 1]) - np.amin(vertices[:, 1])
        ax2_size = np.amax(vertices[:, 2]) - np.amin(vertices[:, 2])

        if self.args.dataset == 'waymo':
            scale_ax0 = self.cfg.templates.template_length / ax0_size
            scale_ax1 = self.cfg.templates.template_width / ax1_size
            scale_ax2 = self.cfg.templates.template_height / ax2_size
        else:
            scale_ax0 = self.cfg.templates.template_width / ax0_size
            scale_ax1 = self.cfg.templates.template_height / ax1_size
            scale_ax2 = self.cfg.templates.template_length / ax2_size

        vertices[:, 0] *= scale_ax0
        vertices[:, 1] *= scale_ax1
        vertices[:, 2] *= scale_ax2

        mesh.vertices = open3d.utility.Vector3dVector(vertices)

        mesh.compute_vertex_normals()  # Looks better in visu with normals
        pcd = mesh.sample_points_uniformly(number_of_points=1000)

        return pcd, mesh

    def load_and_sample_passat_scale(self):
        mesh = open3d.io.read_triangle_mesh("../data/passat_scale.gltf")  # Read mesh of fiat uno converted via blender
        bbox = mesh.get_minimal_oriented_bounding_box()
        T = np.eye(4)
        T[:3, 3] = (-bbox.center[0], -bbox.center[1], -bbox.center[2])
        mesh.transform(T)

        if self.args.dataset == 'waymo':
            T = np.eye(4)
            T[:3, :3] = open3d.geometry.get_rotation_matrix_from_zxy((-np.pi/2, np.pi/2, 0))  # Y rotation -||-
            mesh.transform(T)
        else:
            T = np.eye(4)
            T[:3, :3] = open3d.geometry.get_rotation_matrix_from_zxy((np.pi, 0, np.pi))  # Y rotation -||-
            mesh.transform(T)

        vertices = np.asarray(mesh.vertices)
        ax0_size = np.amax(vertices[:, 0]) - np.amin(vertices[:, 0])
        ax1_size = np.amax(vertices[:, 1]) - np.amin(vertices[:, 1])
        ax2_size = np.amax(vertices[:, 2]) - np.amin(vertices[:, 2])

        if self.args.dataset == 'waymo':
            scale_ax0 = self.cfg.templates.template_length / ax0_size
            scale_ax1 = self.cfg.templates.template_width / ax1_size
            scale_ax2 = self.cfg.templates.template_height / ax2_size
        else:
            scale_ax0 = self.cfg.templates.template_width / ax0_size
            scale_ax1 = self.cfg.templates.template_height / ax1_size
            scale_ax2 = self.cfg.templates.template_length / ax2_size

        vertices[:, 0] *= scale_ax0
        vertices[:, 1] *= scale_ax1
        vertices[:, 2] *= scale_ax2

        mesh.vertices = open3d.utility.Vector3dVector(vertices)

        mesh.compute_vertex_normals()  # Looks better in visu with normals
        pcd = mesh.sample_points_uniformly(number_of_points=1000)

        return pcd, mesh

    def load_and_sample_cube(self):
        mesh = open3d.io.read_triangle_mesh("../data/cube.gltf")  # Read mesh of fiat uno converted via blender
        bbox = mesh.get_minimal_oriented_bounding_box()
        T = np.eye(4)
        T[:3, 3] = (-bbox.center[0], -bbox.center[1], -bbox.center[2])
        mesh.transform(T)

        if self.args.dataset == 'waymo':
            T = np.eye(4)
            T[:3, :3] = open3d.geometry.get_rotation_matrix_from_zxy((0, np.pi/2, 0))  # Y rotation -||-
            mesh.transform(T)
        else:
            T = np.eye(4)
            T[:3, :3] = open3d.geometry.get_rotation_matrix_from_zxy((-np.pi / 2, -np.pi / 2, 0))  # Y rotation -||-
            mesh.transform(T)

        vertices = np.asarray(mesh.vertices)
        ax0_size = np.amax(vertices[:, 0]) - np.amin(vertices[:, 0])
        ax1_size = np.amax(vertices[:, 1]) - np.amin(vertices[:, 1])
        ax2_size = np.amax(vertices[:, 2]) - np.amin(vertices[:, 2])

        if self.args.dataset == 'waymo':
            scale_ax0 = self.cfg.templates.template_length / ax0_size
            scale_ax1 = self.cfg.templates.template_width / ax1_size
            scale_ax2 = self.cfg.templates.template_height / ax2_size
        else:
            scale_ax0 = self.cfg.templates.template_width / ax0_size
            scale_ax1 = self.cfg.templates.template_height / ax1_size
            scale_ax2 = self.cfg.templates.template_length / ax2_size

        vertices[:, 0] *= scale_ax0
        vertices[:, 1] *= scale_ax1
        vertices[:, 2] *= scale_ax2

        mesh.vertices = open3d.utility.Vector3dVector(vertices)

        mesh.compute_vertex_normals()  # Looks better in visu with normals
        pcd = mesh.sample_points_uniformly(number_of_points=1000)

        return pcd, mesh

    def load_and_sample_cube_scale(self):
        mesh = open3d.io.read_triangle_mesh("../data/cube_top.gltf")  # Read mesh of fiat uno converted via blender
        bbox = mesh.get_minimal_oriented_bounding_box()
        T = np.eye(4)
        T[:3, 3] = (-bbox.center[0], -bbox.center[1], -bbox.center[2])
        mesh.transform(T)

        if self.args.dataset == 'waymo':
            T = np.eye(4)
            T[:3, :3] = open3d.geometry.get_rotation_matrix_from_zxy((0, np.pi/2, 0))  # Y rotation -||-
            mesh.transform(T)
        else:
            T = np.eye(4)
            T[:3, :3] = open3d.geometry.get_rotation_matrix_from_zxy((-np.pi / 2, -np.pi / 2, 0))  # Y rotation -||-
            mesh.transform(T)

        vertices = np.asarray(mesh.vertices)
        ax0_size = np.amax(vertices[:, 0]) - np.amin(vertices[:, 0])
        ax1_size = np.amax(vertices[:, 1]) - np.amin(vertices[:, 1])
        ax2_size = np.amax(vertices[:, 2]) - np.amin(vertices[:, 2])

        if self.args.dataset == 'waymo':
            scale_ax0 = self.cfg.templates.template_length / ax0_size
            scale_ax1 = self.cfg.templates.template_width / ax1_size
            scale_ax2 = self.cfg.templates.template_height / ax2_size
        else:
            scale_ax0 = self.cfg.templates.template_width / ax0_size
            scale_ax1 = self.cfg.templates.template_height / ax1_size
            scale_ax2 = self.cfg.templates.template_length / ax2_size

        vertices[:, 0] *= scale_ax0
        vertices[:, 1] *= scale_ax1
        vertices[:, 2] *= scale_ax2

        mesh.vertices = open3d.utility.Vector3dVector(vertices)

        mesh.compute_vertex_normals()  # Looks better in visu with normals
        pcd = mesh.sample_points_uniformly(number_of_points=1000)

        return pcd, mesh

    def load_and_sample_suv(self):
        mesh = open3d.io.read_triangle_mesh("../data/suv.gltf")  # Read mesh of fiat uno converted via blender
        bbox = mesh.get_minimal_oriented_bounding_box()
        T = np.eye(4)
        T[:3, 3] = (-bbox.center[0], -bbox.center[1], -bbox.center[2])
        mesh.transform(T)

        if self.args.dataset == 'waymo':
            T = np.eye(4)
            T[:3, :3] = open3d.geometry.get_rotation_matrix_from_zxy((np.pi/2, np.pi/2, 0))  # Y rotation -||-
            mesh.transform(T)
        else:
            T = np.eye(4)
            T[:3, :3] = open3d.geometry.get_rotation_matrix_from_zxy((np.pi, 0, 0))  # Y rotation -||-
            mesh.transform(T)

        vertices = np.asarray(mesh.vertices)
        ax0_size = np.amax(vertices[:, 0]) - np.amin(vertices[:, 0])
        ax1_size = np.amax(vertices[:, 1]) - np.amin(vertices[:, 1])
        ax2_size = np.amax(vertices[:, 2]) - np.amin(vertices[:, 2])

        if self.args.dataset == 'waymo':
            scale_ax0 = self.cfg.templates.template_length / ax0_size
            scale_ax1 = self.cfg.templates.template_width / ax1_size
            scale_ax2 = self.cfg.templates.template_height / ax2_size
        else:
            scale_ax0 = self.cfg.templates.template_width / ax0_size
            scale_ax1 = self.cfg.templates.template_height / ax1_size
            scale_ax2 = self.cfg.templates.template_length / ax2_size

        vertices[:, 0] *= scale_ax0
        vertices[:, 1] *= scale_ax1
        vertices[:, 2] *= scale_ax2

        mesh.vertices = open3d.utility.Vector3dVector(vertices)

        mesh.compute_vertex_normals()  # Looks better in visu with normals
        pcd = mesh.sample_points_uniformly(number_of_points=1000)

        return pcd, mesh

    def load_and_sample_mpv(self):
        mesh = open3d.io.read_triangle_mesh("../data/minivan.gltf")  # Read mesh of fiat uno converted via blender
        bbox = mesh.get_minimal_oriented_bounding_box()
        T = np.eye(4)
        T[:3, 3] = (-bbox.center[0], -bbox.center[1], -bbox.center[2])
        mesh.transform(T)

        if self.args.dataset == 'waymo':
            T = np.eye(4)
            T[:3, :3] = open3d.geometry.get_rotation_matrix_from_zxy((np.pi, np.pi/2, 0))  # Y rotation -||-
            mesh.transform(T)
        else:
            T = np.eye(4)
            T[:3, :3] = open3d.geometry.get_rotation_matrix_from_zxy((np.pi, 0, np.pi/2))  # Y rotation -||-
            mesh.transform(T)

        vertices = np.asarray(mesh.vertices)
        ax0_size = np.amax(vertices[:, 0]) - np.amin(vertices[:, 0])
        ax1_size = np.amax(vertices[:, 1]) - np.amin(vertices[:, 1])
        ax2_size = np.amax(vertices[:, 2]) - np.amin(vertices[:, 2])

        if self.args.dataset == 'waymo':
            scale_ax0 = self.cfg.templates.template_length / ax0_size
            scale_ax1 = self.cfg.templates.template_width / ax1_size
            scale_ax2 = self.cfg.templates.template_height / ax2_size
        else:
            scale_ax0 = self.cfg.templates.template_width / ax0_size
            scale_ax1 = self.cfg.templates.template_height / ax1_size
            scale_ax2 = self.cfg.templates.template_length / ax2_size

        vertices[:, 0] *= scale_ax0
        vertices[:, 1] *= scale_ax1
        vertices[:, 2] *= scale_ax2

        mesh.vertices = open3d.utility.Vector3dVector(vertices)

        mesh.compute_vertex_normals()  # Looks better in visu with normals
        pcd = mesh.sample_points_uniformly(number_of_points=1000)

        return pcd, mesh

    def prepare_pic(self):
        self.pic = self.pics[self.pic_index]

        temp = self.pic.split("/")
        self.file_name = temp[-1].split(".")[0]

        self.img_orig = cv2.imread(self.pic)
        img = np.array(self.img_orig, dtype=np.uint8)
        self.img = np.moveaxis(img, -1, 0)  # the model expects the image to be in channel first format

        if not self.cfg.general.supress_debug_prints:
            print(self.mapping_data[int(self.random_indexes[int(self.pic_index)]) - 1])

    def prepare_pic_waymo(self, data):
        # First get the name of the file. Sometimes for debug we want to choose it randomly
        if self.cfg.visualization.show_3D_scan:
            self.bboxes = []

        self.img = []
        images_sorted = sorted(data.images, key=lambda i: i.name)
        for index, image in enumerate(images_sorted):
            decoded_image = tf.image.decode_jpeg(image.image).numpy()

            # Open the image, convert
            img = np.array(decoded_image, dtype=np.uint8)
            self.img.append(np.moveaxis(img, -1, 0))  # the model expects the image to be in channel first format

    def compute_mean(self, lidar):
        x_mean = np.median(lidar[:, 0])
        y_mean = np.median(lidar[:, 1])
        z_mean = np.median(lidar[:, 2])

        return x_mean, y_mean, z_mean

    def icp_point_to_plane_open3d(self, source_points, target_points, max_iterations=50, tolerance=1e-6):
        # Convert numpy arrays to Open3D point clouds
        source_cloud = open3d.geometry.PointCloud()
        source_cloud.points = open3d.utility.Vector3dVector(source_points)
        source_cloud.estimate_normals(search_param=open3d.geometry.KDTreeSearchParamHybrid(radius=.5, max_nn=30))
        #source_cloud.estimate_covariances(search_param=open3d.geometry.KDTreeSearchParamHybrid(radius=1., max_nn=30))
        target_cloud = open3d.geometry.PointCloud()
        target_cloud.points = open3d.utility.Vector3dVector(target_points)
        target_cloud.estimate_normals(search_param=open3d.geometry.KDTreeSearchParamHybrid(radius=.5, max_nn=30))
        #target_cloud.estimate_covariances(search_param=open3d.geometry.KDTreeSearchParamHybrid(radius=1., max_nn=30))

        # Perform Point-to-Plane ICP registration
        icp_result = open3d.pipelines.registration.registration_icp(
            source_cloud, target_cloud, 0.1, np.eye(4),
            open3d.pipelines.registration.TransformationEstimationPointToPlane())

        # Get the transformation matrix from the ICP result
        transformation = icp_result.transformation

        return transformation

    def icp_point_to_point_open3d(self, source_points, target_points, max_iterations=50, tolerance=1e-6):
        # Convert numpy arrays to Open3D point clouds
        source_cloud = open3d.geometry.PointCloud()
        source_cloud.points = open3d.utility.Vector3dVector(source_points)
        #source_cloud.estimate_normals(search_param=open3d.geometry.KDTreeSearchParamHybrid(radius=1., max_nn=30))
        #source_cloud.estimate_covariances(search_param=open3d.geometry.KDTreeSearchParamHybrid(radius=2., max_nn=30))
        target_cloud = open3d.geometry.PointCloud()
        target_cloud.points = open3d.utility.Vector3dVector(target_points)
        #target_cloud.estimate_normals(search_param=open3d.geometry.KDTreeSearchParamHybrid(radius=1., max_nn=30))
        #target_cloud.estimate_covariances(search_param=open3d.geometry.KDTreeSearchParamHybrid(radius=2., max_nn=30))

        # Perform Point-to-Plane ICP registration
        icp_result = open3d.pipelines.registration.registration_icp(
            source_cloud, target_cloud, 0.1, np.eye(4),
            open3d.pipelines.registration.TransformationEstimationPointToPoint())

        # Get the transformation matrix from the ICP result
        transformation = icp_result.transformation

        return transformation
