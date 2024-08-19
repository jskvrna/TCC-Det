import copy

import pytorch3d
import torch
import sys
import os
from pytorch3d.io import load_objs_as_meshes, load_obj, save_obj
import pykitti
import matplotlib.pyplot as plt
import numpy as np
import open3d

# Data structures and functions for rendering
from pytorch3d.structures import Meshes, join_meshes_as_scene, join_meshes_as_batch
from pytorch3d.vis.plotly_vis import AxisArgs, plot_batch_individually, plot_scene
from pytorch3d.vis.texture_vis import texturesuv_image_matplotlib
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
    PointLights,
    DirectionalLights,
    Materials,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    TexturesUV,
    TexturesVertex,
    PerspectiveCameras,
    BlendParams,
    HardPhongShader
)
from pytorch3d.transforms import Transform3d, euler_angles_to_matrix, matrix_to_euler_angles

from pytorch3d.renderer.mesh.shader import (HardDepthShader, SoftSilhouetteShader, SoftDepthShader)

sys.path.append(os.path.abspath(''))

from base_class import BaseClass


class Renderer(BaseClass):
    def __init__(self, config_path):
        super().__init__(config_path)
        self.device = torch.device("cuda")
        self.kittiavg = torch.tensor([self.cfg.templates.template_length, self.cfg.templates.template_width, self.cfg.templates.template_height], dtype=torch.float32, device=self.device)
        self.load_templates_renderer()

    def load_templates_renderer(self):
        verts, faces_idx, _ = load_obj(self.fiat_template_obj, device=self.device)
        verts_passat, faces_idx_passat, _ = load_obj(self.passat_template_obj, device=self.device)

        faces = faces_idx.verts_idx
        faces_passat = faces_idx_passat.verts_idx
        verts_rgb = torch.ones_like(verts)[None]  # (1, V, 3)
        verts_rgb_passat = torch.ones_like(verts_passat)[None]  # (1, V, 3)
        textures = TexturesVertex(verts_features=verts_rgb.to(self.device))
        textures_passat = TexturesVertex(verts_features=verts_rgb_passat.to(self.device))

        t_matrix = torch.eye(4, device=self.device)
        t_matrix[:3, :3] = euler_angles_to_matrix(torch.tensor([0., np.pi / 2., 0.], dtype=torch.float32), "XYZ").cuda()
        # t_matrix[:3, :3] = euler_angles_to_matrix(torch.tensor([0., 0., 0.], dtype=torch.float32), "XYZ").cuda()
        transform = Transform3d(device=self.device, matrix=t_matrix)
        verts = transform.transform_points(verts)
        t_matrix = torch.eye(4, device=self.device)
        t_matrix[:3, :3] = euler_angles_to_matrix(torch.tensor([-np.pi / 2., 0., 0.], dtype=torch.float32),
                                                  "XYZ").cuda()
        transform = Transform3d(device=self.device, matrix=t_matrix)
        verts_passat = transform.transform_points(verts_passat)

        self.fiat_mesh = Meshes(
            verts=[verts.to(self.device)],
            faces=[faces.to(self.device)],
            textures=textures
        )

        self.passat_mesh = Meshes(
            verts=[verts_passat.to(self.device)],
            faces=[faces_passat.to(self.device)],
            textures=textures_passat
        )

        bbox = self.fiat_mesh.get_bounding_boxes()
        bbox_passat = self.passat_mesh.get_bounding_boxes()
        center_x = (bbox[0, 0, 0] + bbox[0, 0, 1]) / 2.
        center_y = (bbox[0, 1, 0] + bbox[0, 1, 1]) / 2.
        center_z = (bbox[0, 2, 0] + bbox[0, 2, 1]) / 2.
        center_x_passat = (bbox_passat[0, 0, 0] + bbox_passat[0, 0, 1]) / 2.
        center_y_passat = (bbox_passat[0, 1, 0] + bbox_passat[0, 1, 1]) / 2.
        center_z_passat = (bbox_passat[0, 2, 0] + bbox_passat[0, 2, 1]) / 2.

        length = (bbox[0, 0, 1] - bbox[0, 0, 0])
        width = (bbox[0, 1, 1] - bbox[0, 1, 0])
        height = (bbox[0, 2, 1] - bbox[0, 2, 0])
        length_passat = (bbox_passat[0, 0, 1] - bbox_passat[0, 0, 0])
        width_passat = (bbox_passat[0, 1, 1] - bbox_passat[0, 1, 0])
        height_passat = (bbox_passat[0, 2, 1] - bbox_passat[0, 2, 0])

        print("Fiat dim: ", length, width, height)
        print("Passat dim: ", length_passat, width_passat, height_passat)

        self.fiat_mesh = self.fiat_mesh.offset_verts(
            torch.tensor([-center_x, -center_y, -center_z], device=self.device))
        self.passat_mesh = self.passat_mesh.offset_verts(
            torch.tensor([-center_x_passat, -center_y_passat, -center_z_passat], device=self.device))

        '''
        save_obj("/home/potoso/2D_to_3D_Annotations/3d/data/test.obj",
                 self.passat_mesh.get_mesh_verts_faces(0)[0], self.passat_mesh.get_mesh_verts_faces(0)[1])
        mesh2 = o3d.io.read_triangle_mesh("/home/potoso/2D_to_3D_Annotations/3d/data/test.obj")
        coord_system = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.)
        o3d.visualization.draw_geometries([mesh2, coord_system])
        '''

    def render_mesh(self):
        images = self.phong_renderer(self.fiat_mesh)
        plt.figure(figsize=(10, 10))
        plt.imshow(images[0, ..., 3].cpu().numpy())
        plt.axis("off")
        plt.show()

    def init_rendering(self, batch_dict, batch_id):
        self.prepare_info(int(batch_dict['frame_id'][batch_id]))
        path_to_calib = self.cfg.paths.merged_frames + batch_dict['frame_id'][batch_id] + "_calib" + ".zstd"
        self.uncompress_calib_data(path_to_calib)

        tmp_K = torch.eye(4)
        tmp_K[:3, :4] = torch.tensor(self.P_rect_00)

        T_final = torch.eye(4, device=self.device, dtype=torch.float32)

        rot_W_to_cam = T_final[:3, :3]
        T_W_to_cam = T_final[:3, 3]

        device = self.device

        if len(batch_dict['masks'][batch_id]) > 0:
            h = batch_dict['masks'][batch_id][0].shape[0]
            w = batch_dict['masks'][batch_id][0].shape[1]
        else:
            return

        cameras = PerspectiveCameras(device=device, R=rot_W_to_cam.unsqueeze(0), T=T_W_to_cam.unsqueeze(0),
                                     focal_length=((tmp_K[0, 0], tmp_K[1, 1]),),
                                     principal_point=((tmp_K[0, 2], tmp_K[1, 2]),),
                                     image_size=((h, w),), in_ndc=False)

        blend_params = BlendParams(sigma=1e-5)
        raster_settings = RasterizationSettings(
            image_size=(h, w),
            faces_per_pixel=20,
            max_faces_per_bin=10000, #TODO Play with this parameter.
            blur_radius=2e-4
        )
        # We can add a point light in front of the object.
        self.silhouette_renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=cameras,
                raster_settings=raster_settings
            ),
            shader=SoftSilhouetteShader(blend_params=blend_params)
        )

        blend_params = BlendParams(sigma=0.)
        raster_settings = RasterizationSettings(
            image_size=(h, w),
            faces_per_pixel=10,
            max_faces_per_bin=10000,  # TODO Play with this parameter.
            blur_radius=0.
        )
        # We can add a point light in front of the object.
        self.silhouette_renderer_sharp = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=cameras,
                raster_settings=raster_settings
            ),
            shader=SoftSilhouetteShader(blend_params=blend_params)
        )

        raster_settings = RasterizationSettings(
            image_size=(h, w),
            faces_per_pixel=5,
            max_faces_per_bin=10000  # TODO Play with this parameter.
        )
        # We can add a point light in front of the object.
        lights = PointLights(device=device, location=((2.0, 2.0, -2.0),))
        self.phong_renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=cameras,
                raster_settings=raster_settings
            ),
            shader=SoftDepthShader(device=self.device, cameras=cameras, lights=lights)
        )

    def init_rendering_waymo(self, batch_dict, batch):
        device = self.device
        self.silhouette_renderer = []
        self.silhouette_renderer_sharp = []
        self.phong_renderer = []

        #For all cameras
        for i in range(5):
            h = int(batch_dict['cam_size'][batch][i][0])
            w = int(batch_dict['cam_size'][batch][i][1])

            T_final = torch.eye(4, device=self.device, dtype=torch.float32)

            rot_W_to_cam = T_final[:3, :3]
            T_W_to_cam = T_final[:3, 3]

            intrinsic = torch.tensor(batch_dict['cam_calib'][batch][i], device=self.device, dtype=torch.float32)

            cameras = PerspectiveCameras(device=device, R=rot_W_to_cam.unsqueeze(0), T=T_W_to_cam.unsqueeze(0),
                                         focal_length=((intrinsic[0], intrinsic[1]),),
                                         principal_point=((intrinsic[2], intrinsic[3]),),
                                         image_size=((h, w),), in_ndc=False)

            blend_params = BlendParams(sigma=1e-5)
            raster_settings = RasterizationSettings(
                image_size=(h, w),
                faces_per_pixel=20,
                max_faces_per_bin=10000, #TODO Play with this parameter.
                blur_radius=2e-4
            )
            # We can add a point light in front of the object.
            self.silhouette_renderer.append(MeshRenderer(
                rasterizer=MeshRasterizer(
                    cameras=cameras,
                    raster_settings=raster_settings
                ),
                shader=SoftSilhouetteShader(blend_params=blend_params)
            ))

            blend_params = BlendParams(sigma=0.)
            raster_settings = RasterizationSettings(
                image_size=(h, w),
                faces_per_pixel=10,
                max_faces_per_bin=10000,  # TODO Play with this parameter.
                blur_radius=0.
            )
            # We can add a point light in front of the object.
            self.silhouette_renderer_sharp.append(MeshRenderer(
                rasterizer=MeshRasterizer(
                    cameras=cameras,
                    raster_settings=raster_settings
                ),
                shader=SoftSilhouetteShader(blend_params=blend_params)
            ))

            raster_settings = RasterizationSettings(
                image_size=(h, w),
                faces_per_pixel=5,
                max_faces_per_bin=10000  # TODO Play with this parameter.
            )
            # We can add a point light in front of the object.
            lights = PointLights(device=device, location=((2.0, 2.0, -2.0),))
            self.phong_renderer.append(MeshRenderer(
                rasterizer=MeshRasterizer(
                    cameras=cameras,
                    raster_settings=raster_settings
                ),
                shader=SoftDepthShader(device=self.device, cameras=cameras, lights=lights)
            ))

    def get_target_camera_index(self, batch_dict, bbox, mask_target, batch_idx):
        best_candidate = -1.
        best_score = -1.
        best_mask = None

        for i in range(5):
            mask, _ = self.render_by_single_bbox_waymo(bbox, batch_dict, 0, batch_idx, False, False, i)
            mask_temp = mask[0, :, :, 0]
            if torch.sum(mask_temp) > 1. and mask_target.shape[0] == mask_temp.shape[0] and mask_target.shape[1] == mask_temp.shape[1]:
                score = torch.sum(mask_target * mask_temp)
                if score > best_score:
                    best_score = score
                    best_candidate = i
                    best_mask = mask

        return best_mask, best_mask, best_candidate

    def render_by_single_bbox(self, bbox, data_dict, frame_idx, batch_idx, sharp, template_idx):
        if template_idx == 1:
            mesh = self.passat_mesh.clone()
        else:
            mesh = self.fiat_mesh.clone()

        textures = mesh.textures

        bbox = bbox.clone()

        verts, faces = mesh.get_mesh_verts_faces(0)

        bbox[4] = bbox[4].detach()  # Disable width gradient
        bbox[0] = bbox[0].detach() #Disable X loc gradient
        bbox[1] = bbox[1].detach() #Disable Y loc gradient
        #bbox[2] = bbox[2].detach() #Disable Z gradient
        bbox[3] = bbox[3].detach() #Disable length gradient
        bbox[2] = bbox[2] + 0.05

        scale = bbox[3:6] / self.kittiavg
        verts = verts[:, :] * scale

        r = torch.eye(3, dtype=torch.float32, device=self.device)
        r[0, 0] = torch.cos(bbox[6].detach())
        r[0, 1] = -torch.sin(bbox[6].detach())
        r[1, 0] = torch.sin(bbox[6].detach())
        r[1, 1] = torch.cos(bbox[6].detach())
        verts = torch.matmul(r, verts.T).T

        tmp_mesh = Meshes(
            verts=[verts.to(self.device)],
            faces=[faces.to(self.device)],
            textures=textures
        )

        tmp_mesh = tmp_mesh.offset_verts(bbox[:3])

        #Now lets do inverse data augmentation
        verts, faces = tmp_mesh.get_mesh_verts_faces(0)

        verts_homo = torch.nn.functional.pad(verts, (0,1,0,0), value=1.)

        inv_aug_matrix = torch.linalg.inv(data_dict["lidar_aug_matrix"][batch_idx, :, :]).cuda()

        verts_homo = torch.matmul(inv_aug_matrix, verts_homo.T).squeeze().T

        if 'flip_x' in data_dict and data_dict['flip_x'][batch_idx] > 0.5:
            verts_homo = verts_homo * torch.tensor([1., -1., 1., 1.], dtype=torch.float32, device=self.device)

        T_velo_to_cam = torch.tensor(self.T_cam2_velo, device=self.device, dtype=torch.float32)
        verts_homo = torch.matmul(T_velo_to_cam, verts_homo.T).squeeze().T

        T_cam_to_pytorch3d = torch.eye(4, device=self.device)
        T_cam_to_pytorch3d[:3, :3] = euler_angles_to_matrix(torch.tensor([0., 0., torch.pi], dtype=torch.float32),
                                                            "XYZ").cuda()
        verts_homo = torch.matmul(T_cam_to_pytorch3d, verts_homo.T).squeeze().T

        verts = verts_homo[:, :3]

        tmp_mesh = Meshes(
            verts=[verts.to(self.device)],
            faces=[faces.to(self.device)],
            textures=textures
        )

        if sharp:
            image = self.silhouette_renderer_sharp(tmp_mesh)
        else:
            image = self.silhouette_renderer(tmp_mesh)

        if self.cfg.visualization.show_rendered_picture:
            plt.style.use('dark_background')
            plt.figure(figsize=(10, 10))
            plt.imshow(image[0, ..., 3].cpu().detach().numpy(), cmap='gray')
            plt.axis("off")
            plt.show()

        return image[..., 3].unsqueeze(3), image[..., 3].unsqueeze(3)

    def render_by_single_bbox_waymo(self, bbox, data_dict, frame_idx, batch_idx, sharp, passat, camera):
        if passat:
            mesh = self.passat_mesh.clone()
        else:
            mesh = self.fiat_mesh.clone()

        textures = mesh.textures

        bbox = bbox.clone()

        verts, faces = mesh.get_mesh_verts_faces(0)

        bbox[4] = bbox[4].detach()  # Disable width gradient
        bbox[0] = bbox[0].detach()  # Disable X loc gradient
        bbox[1] = bbox[1].detach()  # Disable Y loc gradient
        # bbox[2] = bbox[2].detach() #Disable Z gradient
        bbox[3] = bbox[3].detach()  # Disable length gradient
        bbox[2] = bbox[2] + 0.05

        scale = bbox[3:6] / self.kittiavg
        verts = verts[:, :] * scale

        r = torch.eye(3, dtype=torch.float32, device=self.device)
        r[0, 0] = torch.cos(bbox[6].detach())
        r[0, 1] = -torch.sin(bbox[6].detach())
        r[1, 0] = torch.sin(bbox[6].detach())
        r[1, 1] = torch.cos(bbox[6].detach())
        verts = torch.matmul(r, verts.T).T

        tmp_mesh = Meshes(
            verts=[verts.to(self.device)],
            faces=[faces.to(self.device)],
            textures=textures
        )

        tmp_mesh = tmp_mesh.offset_verts(bbox[:3])

        #save_obj("/home/potoso/2D_to_3D_Annotations/3d/data/test_mesh.obj",
        #         tmp_mesh.get_mesh_verts_faces(0)[0], tmp_mesh.get_mesh_verts_faces(0)[1])
        #templ1 = open3d.io.read_triangle_mesh("/home/potoso/2D_to_3D_Annotations/3d/data/test_mesh.obj")

        # Now lets do inverse data augmentation
        verts, faces = tmp_mesh.get_mesh_verts_faces(0)

        verts_homo = torch.nn.functional.pad(verts, (0, 1, 0, 0), value=1.)

        inv_aug_matrix = torch.linalg.inv(data_dict["lidar_aug_matrix"][batch_idx, :, :]).cuda()

        verts_homo = torch.matmul(inv_aug_matrix, verts_homo.T).squeeze().T

        if 'flip_y' in data_dict and data_dict['flip_y'][batch_idx] > 0.5:
            verts_homo = verts_homo * torch.tensor([-1., 1., 1., 1.], dtype=torch.float32, device=self.device)
        if 'flip_x' in data_dict and data_dict['flip_x'][batch_idx] > 0.5:
            verts_homo = verts_homo * torch.tensor([1., -1., 1., 1.], dtype=torch.float32, device=self.device)


        T_velo_to_cam = torch.tensor(data_dict['cam_transformation'][batch_idx][camera], device=self.device, dtype=torch.float32)
        T_velo_to_cam = torch.reshape(T_velo_to_cam, (4, 4))

        T_velo_to_cam = torch.linalg.inv(T_velo_to_cam)

        verts_homo = torch.matmul(T_velo_to_cam, verts_homo.T).squeeze().T

        T_cam_to_pytorch3d = torch.eye(4, device=self.device)
        T_cam_to_pytorch3d[:3, :3] = euler_angles_to_matrix(torch.tensor([torch.pi/2., 0., torch.pi/2.], dtype=torch.float32),
                                                            "XYZ").cuda()
        verts_homo = torch.matmul(T_cam_to_pytorch3d, verts_homo.T).squeeze().T
        T_cam_to_pytorch3d2 = torch.eye(4, device=self.device)
        T_cam_to_pytorch3d2[:3, :3] = euler_angles_to_matrix(
            torch.tensor([0., 0., torch.pi], dtype=torch.float32),
            "XYZ").cuda()
        verts_homo = torch.matmul(T_cam_to_pytorch3d2, verts_homo.T).squeeze().T

        verts = verts_homo[:, :3]

        tmp_mesh = Meshes(
            verts=[verts.to(self.device)],
            faces=[faces.to(self.device)],
            textures=textures
        )

        if sharp:
            image = self.silhouette_renderer_sharp[camera](tmp_mesh)
        else:
            image = self.silhouette_renderer[camera](tmp_mesh)

        if self.cfg.visualization.show_rendered_picture:
            plt.style.use('dark_background')
            plt.figure(figsize=(10, 10))
            plt.imshow(image[0, ..., 3].cpu().detach().numpy(), cmap='gray')
            plt.axis("off")
            plt.show()

        #xx = torch.sum(image[..., 3])

        return image[..., 3].unsqueeze(3), image[..., 3].unsqueeze(3)

    def create_target_car_mask(self, corresponding_idx, batch_dict, batch_idx):
        det_mask = batch_dict['masks'][batch_idx][corresponding_idx]
        lidar = batch_dict['lidars'][batch_idx][corresponding_idx]
        location, _ = torch.median(lidar, dim=0)
        dist = np.sqrt(location[0].cpu().item() ** 2 + location[1].cpu().item() ** 2)

        target_mask = torch.zeros(det_mask.shape, device=self.device, dtype=torch.float32)
        target_weights = torch.ones(det_mask.shape, device=self.device, dtype=torch.float32)

        #True, there is supposed to be a car
        target_mask[det_mask == True] = 1.

        #Dontcares are all other cars, thus for this region the weight is zero.
        for i in range(len(batch_dict['masks'][batch_idx])):
            if i != corresponding_idx:
                lidar = batch_dict['lidars'][batch_idx][i]
                location, _ = torch.median(lidar, dim=0)
                tmp_dist = np.sqrt(location[0].cpu().item() ** 2 + location[1].cpu().item() ** 2)
                if dist > tmp_dist:
                    target_weights[batch_dict['masks'][batch_idx][i] == True] = 0.
            else:
                continue

        return target_mask, target_weights

    def create_target_car_mask_waymo(self, corresponding_idx, batch_dict, batch_idx):
        moving_car_idx = corresponding_idx - len(batch_dict['standing'][batch_idx])
        # If the car is moving
        if moving_car_idx >= 0:
            det_mask = batch_dict['moving_masks'][batch_idx][0][moving_car_idx]
            if det_mask.shape[1] == 1:
                return None, None
            location = batch_dict['moving'][batch_idx][0][moving_car_idx][0][:2]
            dist = np.sqrt(location[0] ** 2 + location[1] ** 2)
        else:
            det_mask = batch_dict['standing_masks'][batch_idx][corresponding_idx]
            if det_mask.shape[1] == 1:
                return None, None
            lidar = batch_dict['standing'][batch_idx][corresponding_idx]
            location, _ = torch.median(lidar, dim=0)
            dist = np.sqrt(location[0].cpu().item() ** 2 + location[1].cpu().item() ** 2)

        target_mask = torch.zeros(det_mask.shape, device=self.device, dtype=torch.float32)
        target_weights = torch.ones(det_mask.shape, device=self.device, dtype=torch.float32)

        # True, there is supposed to be a car
        target_mask[det_mask == True] = 1.

        return target_mask.T, target_weights.T
        #TODO repair this
        # Dontcares are all other cars, thus for this region the weight is zero.
        standing_cars_masks = batch_dict['standing_masks'][batch_idx]
        moving_cars_masks = batch_dict['moving_masks'][batch_idx][0]

        if moving_car_idx >= 0:
            for i in range(len(standing_cars_masks)):
                lidar = batch_dict['standing'][batch_idx][i]
                location, _ = torch.median(lidar, dim=0)
                tmp_dist = np.sqrt(location[0].cpu().item() ** 2 + location[1].cpu().item() ** 2)
                if dist > tmp_dist and standing_cars_masks[i].shape[1] != 1:
                    target_weights[standing_cars_masks[i] == True] = 0.

            for i in range(len(moving_cars_masks)):
                if i == moving_car_idx:
                    continue
                location = batch_dict['moving'][batch_idx][0][i][0][:2]
                tmp_dist = np.sqrt(location[0] ** 2 + location[1] ** 2)
                if dist > tmp_dist and moving_cars_masks[i].shape[1] != 1:
                    target_weights[moving_cars_masks[i] == True] = 0.
        else:
            for i in range(len(standing_cars_masks)):
                if i == corresponding_idx:
                    continue
                lidar = batch_dict['standing'][batch_idx][i]
                location, _ = torch.median(lidar, dim=0)
                tmp_dist = np.sqrt(location[0].cpu().item() ** 2 + location[1].cpu().item() ** 2)
                if dist > tmp_dist and standing_cars_masks[i].shape[1] != 1:
                    target_weights[standing_cars_masks[i] == True] = 0.

            for i in range(len(moving_cars_masks)):
                location = batch_dict['moving'][batch_idx][0][i][0][:2]
                tmp_dist = np.sqrt(location[0] ** 2 + location[1] ** 2)
                if dist > tmp_dist and moving_cars_masks[i].shape[1] != 1:
                    target_weights[moving_cars_masks[i] == True] = 0.

        return target_mask, target_weights
