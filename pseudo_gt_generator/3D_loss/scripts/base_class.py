import glob
import os
import yaml

class DictAsObject:
    def __init__(self, data):
        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, dict):
                    self.__dict__[key] = DictAsObject(value)
                elif isinstance(value, list):
                    self.__dict__[key] = [DictAsObject(item) if isinstance(item, dict) else item for item in value]
                else:
                    self.__dict__[key] = value
        else:
            raise TypeError("Expected a dictionary")

    def __getattr__(self, item):
        return self.__dict__.get(item)

    def __repr__(self):
        return f"{self.__dict__}"

class BaseClass:
    cluster = True

    def load_yaml_as_object(self, file_path):
        with open(file_path, 'r') as file:
            data = yaml.safe_load(file)
        return DictAsObject(data)

    def __init__(self, config_path):
        self.cfg = self.load_yaml_as_object(config_path)

        self.pics = glob.glob(self.cfg.paths.kitti + 'object_detection/training/image_2/*.png')
        self.pics = sorted(self.pics, key=os.path.basename)
        self.lidar_path = self.cfg.paths.kitti + "object_detection/training/velodyne/"
        self.calib_path = self.cfg.paths.kitti + "object_detection/training/calib/"
        self.label_path = self.cfg.paths.kitti + "object_detection/training/label_2/"
        self.complete_kitti_path = self.cfg.paths.kitti + 'complete_sequences/'

        self.train_rand = self.cfg.paths.kitti +'object_detection/devkit_object/mapping/train_rand.txt'
        self.train_map = self.cfg.paths.kitti + 'object_detection/devkit_object/mapping/train_mapping.txt'

        self.fiat_template_path = self.cfg.paths.tcc_det + "3d/data/pcloud_filtered/" + "999" + ".pcd"
        self.passat_template_path = self.cfg.paths.tcc_det + "3d/data/pcloud_filtered_passat/" + "999" + ".pcd"
        self.fiat_template_obj = self.cfg.paths.tcc_det + "3d/data/fiat3_voxel.obj"
        self.passat_template_obj = self.cfg.paths.tcc_det + "3d/data/passat_voxel.obj"

        self.lidar_car_template_non_filt = None
        self.file_name = None
        self.file_number = None
        self.path_to_folder = None
        self.map_data_cur = None
        self.standing_cars_lidar = None
        self.dont_cares = None
        self.moving_cars_lidar = None
        self.moving_cars_info = None
        self.moving_cars_angles = None
        self.plotter = None
        self.moving_cars_masks = None
        self.standing_cars_masks = None
        self.device = None
        self.fiat_mesh = None
        self.passat_mesh = None
        self.random_indexes = None
        self.mapping_data = None
        self.renderer = None
        self.rasterizer = None
        self.wall_template = None
        self.T_cam2_velo = None
        self.P_rect_00 = None


