import numpy as np
import pickle
import pyvista as pv
import zstd
import open3d
import torch

from base_class import BaseClass


class Loader(BaseClass):
    def __init__(self, config_path):
        super().__init__(config_path)
        self.random_indexes = []
        self.mapping_data = []
        self.load_templates()

        with open(self.train_rand, 'r') as f:
            line = f.readline().strip()
            self.random_indexes = line.split(',')

        with open(self.train_map, 'r') as f:
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

    def load_frame_data(self, batch_dict, frame_idx):
        frame_idx = int(frame_idx)
        self.prepare_info(frame_idx)
        path_to_cars = self.cfg.paths.merged_frames + str(frame_idx).zfill(6) + ".zstd"
        path_to_calib = self.cfg.paths.merged_frames + str(frame_idx).zfill(6) + "_calib" + ".zstd"

        cars = self.uncompress_data(path_to_cars)
        self.uncompress_calib_data(path_to_calib)

        cars = self.transform_lidars_to_velo(cars)
        cars = self.transform_locations_to_velo(cars)

        #Assign points to GT
        cars = self.convert_locs_to3D(cars)

        lidars, masks, locations, moving = self.extract_data_from_cars(cars)

        batch_dict['lidars'] = lidars
        batch_dict['masks'] = masks
        batch_dict['locations'] = locations
        batch_dict['moving'] = moving

        return batch_dict

    def uncompress_data(self, path):
        with open(path, 'rb') as f:
            decompressed_data = zstd.decompress(f.read())
        cars = pickle.loads(decompressed_data)

        new_cars = []
        for car in cars:
            if car.lidar is not None and car.locations is not None and car.mask is not None and car.optimized:
                new_cars.append(car)

        cars = sorted(new_cars, key=lambda x: x.lidar.shape[0], reverse=True)
        return cars

    def uncompress_calib_data(self, path):
        with open(path, 'rb') as f:
            decompressed_data = zstd.decompress(f.read())
        calib = pickle.loads(decompressed_data)

        self.T_cam2_velo = calib[0]
        self.P_rect_00 = calib[1]

    def transform_lidars_to_velo(self, cars):
        for car in cars:
            car.lidar = self.transform_lidar_cam2_to_velo(car.lidar[:3, :].T)
        return cars

    def transform_locations_to_velo(self, cars):
        for car in cars:
            car.locations = self.transform_location_cam2_to_velo(car.locations)
            car.locations = np.array(car.locations)
        return cars

    def convert_locs_to3D(self, cars):
        T_velo_to_cam2 = self.T_cam2_velo
        T_cam2_to_velo = np.linalg.inv(T_velo_to_cam2)

        for car in cars:
            tmp_loc = np.array([car.x, car.y, car.z])
            tmp_loc = np.matmul(T_cam2_to_velo[:3, :3], tmp_loc)
            tmp_loc += T_cam2_to_velo[:3, 3]

            car.x = tmp_loc[0]
            car.y = tmp_loc[1]
            car.z = tmp_loc[2]

        return cars

    def extract_data_from_cars(self, cars):
        lidars = []
        masks = []
        locations = []
        moving = []

        for car in cars:
            #TODO Remove this if statement
            if isinstance(car.mask, list):
                car.mask = car.mask[0]
            lidars.append(torch.tensor(car.lidar, dtype=torch.float32))
            masks.append(torch.tensor(car.mask.astype(bool), dtype=torch.bool))
            locations.append(torch.tensor(car.locations, dtype=torch.float32))
            moving.append(car.moving)

        moving = torch.tensor(moving, dtype=torch.bool)

        return lidars, masks, locations, moving

    def load_templates(self):
        pcd1, _ = self.load_and_sample_fiat()
        pcd2, _ = self.load_and_sample_passat()
        pcd3, _ = self.load_and_sample_suv()
        pcd4, _ = self.load_and_sample_mpv()

        #coord_frame = open3d.geometry.TriangleMesh.create_coordinate_frame(size=1.)  # Only for visu purpose
        #open3d.visualization.draw_geometries([pcd1, coord_frame])

        pcd1 = np.asarray(pcd1.points)
        pcd2 = np.asarray(pcd2.points)
        pcd3 = np.asarray(pcd3.points)
        pcd4 = np.asarray(pcd4.points)

        if self.cfg.general.dataset == 'waymo':
            pcd1[:, 2] += self.cfg.templates.offset_fiat
            pcd2[:, 2] += self.cfg.templates.offset_passat
            pcd3[:, 2] += self.cfg.templates.offset_suv
            pcd4[:, 2] += self.cfg.templates.offset_mpv
        else:
            pcd1[:, 1] -= self.cfg.templates.offset_fiat
            pcd2[:, 1] -= self.cfg.templates.offset_passat
            pcd3[:, 1] -= self.cfg.templates.offset_suv
            pcd4[:, 1] -= self.cfg.templates.offset_mpv

        if self.cfg.visualization.show_loaded_templates:
            cloud_mesh = pv.PolyData(pcd1)
            cloud_mesh2 = pv.PolyData(pcd2)
            cloud_mesh3 = pv.PolyData(pcd3)
            cloud_mesh4 = pv.PolyData(pcd4)
            plotter = pv.Plotter()
            plotter.add_points(cloud_mesh, color='red', point_size=3)
            plotter.add_points(cloud_mesh2, color='blue', point_size=3)
            plotter.add_points(cloud_mesh3, color='yellow', point_size=3)
            plotter.add_points(cloud_mesh4, color='green', point_size=3)
            origin_point = pv.PolyData([[0, 0, 0]])
            plotter.add_points(origin_point, color='white', point_size=20)
            plotter.add_axes()
            plotter.set_background('black')
            plotter.show()

        self.lidar_car_template_non_filt = [torch.tensor(pcd1, dtype=torch.float32).cuda(), torch.tensor(pcd2, dtype=torch.float32).cuda(),
                                            torch.tensor(pcd3, dtype=torch.float32).cuda(), torch.tensor(pcd4, dtype=torch.float32).cuda()]

    def load_and_sample_fiat(self):
        mesh = open3d.io.read_triangle_mesh(self.cfg.paths.tcc_det + "3d/data/" + "/fiat2.gltf")  # Read mesh of fiat uno converted via blender
        bbox = mesh.get_minimal_oriented_bounding_box()
        T = np.eye(4)
        T[:3, 3] = (-bbox.center[0], -bbox.center[1], -bbox.center[2])
        mesh.transform(T)

        if self.cfg.general.dataset == 'waymo':
            T = np.eye(4)
            T[:3, :3] = open3d.geometry.get_rotation_matrix_from_zxy((np.pi/2, np.pi, 0))  # Y rotation -||-
            mesh.transform(T)
        else:
            T = np.eye(4)
            T[:3, :3] = open3d.geometry.get_rotation_matrix_from_zxy((np.pi, np.pi/2, 0))  # Y rotation -||-
            mesh.transform(T)

        vertices = np.asarray(mesh.vertices)
        ax0_size = np.amax(vertices[:, 0]) - np.amin(vertices[:, 0])
        ax1_size = np.amax(vertices[:, 1]) - np.amin(vertices[:, 1])
        ax2_size = np.amax(vertices[:, 2]) - np.amin(vertices[:, 2])

        if self.cfg.general.dataset == 'waymo':
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
        mesh = open3d.io.read_triangle_mesh(self.cfg.paths.tcc_det + "3d/data/" + "passat2.gltf")  # Read mesh of fiat uno converted via blender
        bbox = mesh.get_minimal_oriented_bounding_box()
        T = np.eye(4)
        T[:3, 3] = (-bbox.center[0], -bbox.center[1], -bbox.center[2])
        mesh.transform(T)

        if self.cfg.general.dataset == 'waymo':
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

        if self.cfg.general.dataset == 'waymo':
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
        mesh = open3d.io.read_triangle_mesh(self.cfg.paths.tcc_det + "3d/data/" + "suv.gltf")  # Read mesh of fiat uno converted via blender
        bbox = mesh.get_minimal_oriented_bounding_box()
        T = np.eye(4)
        T[:3, 3] = (-bbox.center[0], -bbox.center[1], -bbox.center[2])
        mesh.transform(T)

        if self.cfg.general.dataset == 'waymo':
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

        if self.cfg.general.dataset == 'waymo':
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
        mesh = open3d.io.read_triangle_mesh(self.cfg.paths.tcc_det + "3d/data/" + "minivan.gltf")  # Read mesh of fiat uno converted via blender
        bbox = mesh.get_minimal_oriented_bounding_box()
        T = np.eye(4)
        T[:3, 3] = (-bbox.center[0], -bbox.center[1], -bbox.center[2])
        mesh.transform(T)

        if self.cfg.general.dataset == 'waymo':
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

        if self.cfg.general.dataset == 'waymo':
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

    def prepare_info(self, idx):
        self.file_name = str(idx).zfill(6)
        self.map_data_cur = self.mapping_data[int(self.random_indexes[idx]) - 1]
        self.file_number = int(self.map_data_cur[2])
        self.path_to_folder = self.complete_kitti_path + self.map_data_cur[0] + '/' + self.map_data_cur[1] + '/'

