from loader import Loader
from filtering import Filtering
from visualization import Visualization
from optimizer import Optimizer
from loss import Losses
from output import Output
from scale_detector import ScaleDetector
from context_growing import CAARGrowing
from stitching import Stitching
from tracker_ODTrack import Tracker_ODTrack
from custom_dataset import CustomDataset

import os
import zstd
import pickle
import pykitti
import copy

class DetAndTracking:
    def __init__(self, lidar_points, lidar_locations, lidar_info, masks):
        self.lidar_points = lidar_points
        self.lidar_locations = lidar_locations
        self.lidar_info = lidar_info
        self.masks = masks

class MainClass(Output, Losses, Optimizer, Visualization, Filtering, Loader, ScaleDetector, CAARGrowing, Stitching, Tracker_ODTrack, CustomDataset):
    def __init__(self, args):
        super().__init__(args)

    def main_waymo(self, argv):
        if self.generate_raw_masks_or_tracking:
            self.load_det2_and_sam()
        seq_num_min, seq_num_max = self.limit_sequences()

        self.prepare_dirs()

        self.load_lidar_templatesv2()
        dict_idx_to_opt = self.load_idx_to_opt()

        for self.segment_index in range(seq_num_min, seq_num_max):
            if not self.cfg.general.supress_debug_prints:
                print("Segment Index: ", self.segment_index)
            if self.do_single_segment is not None and self.do_single_segment != self.segment_index:
                continue

            self.load_current_segment()

            if self.generate_raw_lidar:
                continue

            if not self.load_merged_frames and not self.generate_transformations_only:
                if self.cfg.frames_creation.tracker_for_merging == '2D' and self.args.dataset == 'waymo':
                    if self.check_for_merging_done(dict_idx_to_opt):
                        continue
                    if not self.perform_stitching_and_trackingV2():
                        continue
                else:
                    if not self.perform_mask_extraction():
                        continue

            for self.pic_index, data in enumerate(self.waymo_data):
                if self.cfg.dataset.dataset_stride > 1 and not self.generate_transformations_only:
                    segment_name = self.random_indexes[self.segment_index].split('.')[0]
                    tmp_idx_arr = dict_idx_to_opt[segment_name]
                    if self.pic_index not in tmp_idx_arr:
                        continue
                if self.cfg.custom_dataset.create_custom_dataset:
                    if self.pic_index <= 80 or self.pic_index >= 120:
                        continue

                if not self.cfg.general.supress_debug_prints:
                    print("Frame Index: ", self.pic_index)
                self.prepare_pic_waymo(self.waymo_frame[self.pic_index])

                if self.check_for_optim_done() and self.cfg.output.output_txt:
                    continue

                if self.generate_merged_frames_only:
                    if self.cfg.frames_creation.tracker_for_merging == '2D':
                        self.load_and_prepare_lidar_scan_from_multiple_waymo_tracker(save=True)
                    else:
                        self.load_and_prepare_lidar_scan_from_multiple_waymo(save=True)
                    continue
                elif self.generate_transformations_only:
                    self.calculate_transformations_waymo(self.cfg.frames_creation.nscans_transformation_range, self.cfg.frames_creation.nscans_transformation_range, save=True)
                    continue
                elif self.load_merged_frames:
                    if self.cfg.frames_creation.tracker_for_merging == '2D':
                        self.load_merged_frames_from_files_waymo_tracker(track2D=True, merge_two_trackers=self.cfg.optimization.merge_two_trackers)
                    else:
                        self.load_merged_frames_from_files_waymo_tracker(track2D=False, merge_two_trackers=self.cfg.optimization.merge_two_trackers)
                else:
                    if self.cfg.frames_creation.tracker_for_merging == '2D':
                        self.load_and_prepare_lidar_scan_from_multiple_waymo_tracker(save=False)
                    else:
                        self.load_and_prepare_lidar_scan_from_multiple_waymo(save=False)

                if self.do_optim:
                    if not self.cfg.general.supress_debug_prints:
                        print("Optimizing over: ", len(self.cars), " cars")
                    for car_idx in range(len(self.cars)):
                        if not self.est_location_and_downsample(self.cars[car_idx]):
                            continue

                        self.cars[car_idx] = self.optimize_car(self.cars[car_idx])

                if not self.cfg.general.supress_debug_prints:
                    print("Doing NMS")

                self.cars, to_be_reopt = self.non_maxima_surpression(self.cars)
                if self.cfg.optimization.nms_merge_and_reopt:
                    if not self.cfg.general.supress_debug_prints:
                        print("Doing NMS merge and reopt")
                        print("Optimizing over: ", len(to_be_reopt), " cars")
                    for index in to_be_reopt:
                        if not self.est_location_and_downsample(self.cars[index]):
                            continue
                        self.cars[index] = self.optimize_car(self.cars[index])

                if self.cfg.scale_detector.use_scale_detector and self.do_optim:
                    self.cars = self.extract_lidar_data_from_bbox_tracker(self.cars)
                    if self.cfg.custom_dataset.create_custom_dataset:
                        self.create_custom_dataset_from_cars()
                    if self.do_optim_scale:
                        for i in range(len(self.cars)):
                            self.cars[i] = self.optimize_car_scale(self.cars[i])

                        if self.cfg.scale_detector.use_bbox_reducer:
                            self.bbox_reducer_tracked(self.cars)

                if self.cfg.output.output_txt:
                    self.writetxt_cars(self.cars)

                if self.cfg.visualization.show_3D_scan:
                    self.visualize_3D(self.cars)
                if self.cfg.visualization.show_image:
                    self.show_image(self.img)

                if not self.cfg.general.supress_debug_prints:
                    print("File name: ", self.file_name)
                    print("Pics_done: ", self.pic_index)

    def main_kitti(self, argv):
        if self.generate_raw_masks_or_tracking:
            self.load_det2_and_sam()

        self.prepare_dirs()
        pic_num_min, pic_num_max = self.limit_sequences()

        self.load_lidar_templatesv2()

        for self.pic_index in range(pic_num_min, pic_num_max):
            if not self.cfg.general.supress_debug_prints:
                print("Pic Index: ", self.pic_index)
            self.prepare_pic()
            map_data_cur = self.prepare_frame()

            if self.generate_raw_masks_or_tracking:
                self.precompute_detectron_kitti()
                continue

            if self.cfg.output.output_txt:
                if not self.cfg.general.supress_debug_prints:
                    print("Checking for opened file")
                if os.path.isfile(self.cfg.paths.merged_frames_path + self.file_name + '.txt'):
                    continue

            if self.generate_merged_frames_only:
                self.load_and_prepare_lidar_scan_from_multiple_pykittiV2(self.file_name, self.img, save=True)
                continue
            elif self.generate_transformations_only:
                if os.path.isfile(self.cfg.paths.merged_frames_path + "/transformations/" + self.file_name + '.npy'):
                    continue
                if self.cfg.frames_creation.use_icp:
                    self.calculate_transformationsV2(self.cfg.frames_creation.nscans_transformation_range,
                                                     self.cfg.frames_creation.nscans_transformation_range, save=True)
                else:
                    self.calculate_transformations(self.cfg.frames_creation.nscans_transformation_range,
                                                   self.cfg.frames_creation.nscans_transformation_range, save=True)
                continue
            elif self.load_merged_frames:
                self.load_merged_frames_from_files_KITTI()
            else:
                self.load_and_prepare_lidar_scan_from_multiple_pykittiV2(self.file_name, self.img, save=False)

            if self.do_optim:
                if not self.cfg.general.supress_debug_prints:
                    print("Optimizing over: ", len(self.cars), " cars")
                for car_idx in range(len(self.cars)):
                    if not self.est_location_and_downsample(self.cars[car_idx]):
                        continue

                    self.cars[car_idx] = self.optimize_car(self.cars[car_idx])

            if not self.cfg.general.supress_debug_prints:
                print("Doing NMS")

            self.cars, to_be_reopt = self.non_maxima_surpression(self.cars)
            if self.cfg.optimization.nms_merge_and_reopt:
                if not self.cfg.general.supress_debug_prints:
                    print("Doing NMS merge and reopt")
                    print("Optimizing over: ", len(to_be_reopt), " cars")
                for index in to_be_reopt:
                    if not self.est_location_and_downsample(self.cars[index]):
                        continue
                    self.cars[index] = self.optimize_car(self.cars[index])

            if self.cfg.scale_detector.use_scale_detector and self.do_optim:
                if not self.cfg.general.supress_debug_prints:
                    print("Aggregating points for scale detector")
                self.cars = self.extract_lidar_data_from_bbox_tracker(self.cars)
                if self.cfg.custom_dataset.create_custom_dataset:
                    self.create_custom_dataset_from_cars()
                else:
                    for i in range(len(self.cars)):
                        if not self.cfg.general.supress_debug_prints:
                            print("Optimizing scale: ", i, " from ", len(self.cars))
                        self.cars[i] = self.optimize_car_scale(self.cars[i])

                    if self.cfg.scale_detector.use_bbox_reducer:
                        self.bbox_reducer_tracked(self.cars)

            if self.cfg.output.output_txt:
                self.writetxt_cars(self.cars)
            if self.cfg.output.save_optimized_cars:
                self.save_optimized_cars(self.cars)

            if self.cfg.visualization.show_3D_scan:
                self.visualize_3D(self.cars)
            if self.cfg.visualization.show_image:
                self.show_image(self.img)

            if not self.cfg.general.supress_debug_prints:
                print("File name: ", self.file_name)
                print("Pics_done: ", self.pic_index)

    def main_custom(self, argv):
        self.args = argv

        self.load_lidar_templatesv2()

        while True:
            self.load_custom_dataset()

            if self.do_optim:
                for car_idx in range(len(self.cars)):
                    if not self.est_location_and_downsample(self.cars[car_idx]):
                        continue

                    self.cars[car_idx] = self.optimize_car(self.cars[car_idx])

            if self.do_optim_scale:
                scale_cars = copy.deepcopy(self.cars)
                scale_cars = self.convert_optim_to_scale(scale_cars)
            else:
                scale_cars = None

            if self.do_optim_scale:
                for car_idx in range(len(scale_cars)):
                    if not self.est_location_and_downsample(scale_cars[car_idx]):
                        continue
                    scale_cars[car_idx] = self.optimize_car_scale(scale_cars[car_idx])

            if not self.cfg.general.supress_debug_prints:
                print("Doing NMS")

            self.custom_compute_iou(self.cars)
            self.custom_compute_iou(scale_cars)

            if self.cfg.visualization.show_3D_scan:
                self.visualize_3D_custom_dataset(self.cars, scale_cars)

            if not self.cfg.general.supress_debug_prints:
                print("File name: ", self.file_name)
                print("Pics_done: ", self.pic_index)

    def prepare_frame(self):
        map_data_cur = self.mapping_data[int(self.random_indexes[int(self.file_name)]) - 1]
        self.kitti_data = pykitti.raw(self.cfg.paths.kitti_path + '/complete_sequences/', map_data_cur[0],map_data_cur[1].split("_")[-2])
        self.file_number = int(map_data_cur[2])
        self.path_to_folder = self.cfg.paths.kitti_path + '/complete_sequences/' + map_data_cur[0] + '/' + map_data_cur[1] + '/'
        return map_data_cur

    def load_det2_and_sam(self):
        if not self.load_merged_frames:
            self.load_and_init_detectron_lazy()
            if self.cfg.frames_creation.use_SAM:
                self.load_and_init_SAM() #TODO repair this

    def limit_sequences(self):
        seg_num_max = len(self.random_indexes)
        if self.args.seq_start != -1:
            seg_num_min = self.args.seq_start
        else:
            seg_num_min = self.cfg.general.seq_start
        if self.args.seq_end != -1:
            if self.args.seq_end < len(self.random_indexes):
                seg_num_max = self.args.seq_end
        else:
            if self.cfg.general.seq_end < len(self.random_indexes):
                seg_num_max = self.cfg.general.seq_end

        return seg_num_min, seg_num_max

    def perform_stitching_and_trackingV2(self):
        self.stitched_imgs = []
        self.stitched_imgs, self.homos_all = self.perform_img_stitching()

        if self.generate_homographies_only:
            return False

        if not self.cfg.general.supress_debug_prints:
            print("Images stitched")
        if self.generate_raw_masks_or_tracking:
            pred_masks, _, mask_idxs = self.perform_tracking(self.stitched_imgs, self.homos_all)
            if not self.cfg.general.supress_debug_prints:
                print("Car tracking done")

            self.save_det_and_trackingV2(pred_masks, mask_idxs)
            return False
        else:
            self.load_det_and_trackingV2()
            return True

    def perform_mask_extraction(self):
        self.prec_detectron_output = self.precompute_detectron_waymo()
        if self.generate_raw_masks_or_tracking:
            return False
        self.precompute_standing_car_candidates_waymo()
        return True

    def check_for_optim_done(self):
        if not self.cfg.general.supress_debug_prints:
            print("Checking for opened file")
        if not os.path.isdir(self.cfg.paths.labels_path):
            os.mkdir(self.cfg.paths.labels_path)
        if not os.path.isdir(self.cfg.paths.labels_path + self.file_name):
            os.mkdir(self.cfg.paths.labels_path + self.file_name)
        if os.path.isfile(self.cfg.paths.labels_path + self.file_name + "/" + str(self.pic_index) + '.txt'):
            return True
        else:
            return False

    def load_idx_to_opt(self):
        if self.cfg.dataset.dataset_stride > 1:
            idx_dict = {}
            #Open a txt file, read all lines, separate by ;
            with open("../data/waymo_infos_20_downsampled.txt", 'r') as f:
                lines = f.readlines()

                for line in lines:
                    elements = line.strip().split(';')
                    segment_name = elements.pop(0)  # remove the first element
                    int_elements = [int(e) for e in elements if e]  # convert the rest to integers
                    idx_dict[str(segment_name)] = int_elements

            return idx_dict
        else:
            return None

    def save_det_and_trackingV2(self, pred_masks, mask_idxs):
        if not os.path.isdir(self.cfg.paths.merged_frames_path + "detandtrackedV2/" + self.file_name):
            os.mkdir(self.cfg.paths.merged_frames_path + "detandtrackedV2/" + self.file_name)

        idx = 0
        for mask in pred_masks:
            compressed_arr = zstd.compress(pickle.dumps(mask, pickle.HIGHEST_PROTOCOL))
            with open(self.cfg.paths.merged_frames_path + "detandtrackedV2/" + self.file_name + "/" + "mask_" + str(idx) + ".zstd", 'wb') as f:
                f.write(compressed_arr)
            idx += 1

        idx = 0
        for info in mask_idxs:
            compressed_arr = zstd.compress(pickle.dumps(info, pickle.HIGHEST_PROTOCOL))
            with open(self.cfg.paths.merged_frames_path + "detandtrackedV2/" + self.file_name + "/" + "info_" + str(idx) + ".zstd", 'wb') as f:
                f.write(compressed_arr)
            idx += 1

    def load_det_and_trackingV2(self):
        pred_masks = []
        extracted_info = []

        idx = 0
        while True:
            if os.path.exists(
                    self.cfg.paths.merged_frames_path + "detandtrackedV2/" + self.file_name + "/" + "mask_" + str(idx) + ".zstd"):
                with open(self.cfg.paths.merged_frames_path + "detandtrackedV2/" + self.file_name + "/" + "mask_" + str(idx) + ".zstd",
                          'rb') as f:
                    idx += 1
                    decompressed = zstd.decompress(f.read())
                    mask = pickle.loads(decompressed)
                    pred_masks.append(mask)
            else:
                break

        idx = 0
        while True:
            if os.path.exists(
                    self.cfg.paths.merged_frames_path + "detandtrackedV2/" + self.file_name + "/" + "info_" + str(
                        idx) + ".zstd"):
                with open(self.cfg.paths.merged_frames_path + "detandtrackedV2/" + self.file_name + "/" + "info_" + str(
                        idx) + ".zstd",
                          'rb') as f:
                    idx += 1
                    decompressed = zstd.decompress(f.read())
                    info = pickle.loads(decompressed)
                    extracted_info.append(info)
            else:
                break

        extracted_lidar_points, extracted_lidar_positions, extracted_masks = self.extract_lidar_features(pred_masks, extracted_info)

        self.compressed_detandtracked = []
        for lidar, loc, info, mask in zip(extracted_lidar_points, extracted_lidar_positions, extracted_info, extracted_masks):
            to_save = DetAndTracking(lidar, loc, info, mask)
            compressed_arr = zstd.compress(pickle.dumps(to_save, pickle.HIGHEST_PROTOCOL))
            self.compressed_detandtracked.append(compressed_arr)

    def check_for_merging_done(self, dict_idx_to_opt):
        missing = False
        for pic_index, data in enumerate(self.waymo_data):
            if self.cfg.dataset.dataset_stride > 1:
                segment_name = self.random_indexes[self.segment_index].split('.')[0]
                tmp_idx_arr = dict_idx_to_opt[segment_name]
                if pic_index not in tmp_idx_arr:
                    continue

            if self.cfg.frames_creation.use_growing_for_point_extraction:
                if not os.path.isfile(self.cfg.paths.merged_frames_path + "/cars_2DTrack_growing/" + self.file_name + "/" + str(pic_index) + '.zstd'):
                    missing = True
            else:
                if not os.path.isfile(self.cfg.paths.merged_frames_path + "/cars_2DTrack/" + self.file_name + "/" + str(pic_index) + '.zstd'):
                    missing = True

        return not missing

    def save_optimized_cars(self, cars):
        for car in cars:
            car.bbox = None
        compressed_arr = zstd.compress(pickle.dumps(cars, pickle.HIGHEST_PROTOCOL))

        with open(self.cfg.paths.optimized_cars_path + self.file_name + ".zstd", 'wb') as f:
            f.write(compressed_arr)

        #Save also calib info
        calib = [self.kitti_data.calib.T_cam2_velo, self.kitti_data.calib.P_rect_00]
        compressed_arr = zstd.compress(pickle.dumps(calib, pickle.HIGHEST_PROTOCOL))
        with open(self.cfg.paths.optimized_cars_path + self.file_name + "_calib" + ".zstd", 'wb') as f:
            f.write(compressed_arr)









        

