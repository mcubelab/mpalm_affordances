import os, sys
import pickle
import numpy as np
import trimesh
import open3d
import copy
import time
import argparse
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from airobot.utils import common

from rpo_planning.utils import common as util
from rpo_planning.utils.contact import correct_palm_pos_single, correct_grasp_pos


class PalmVis(object):
    def __init__(self, palm_mesh_file, table_mesh_file, cfg):
        self.palm_mesh_file = palm_mesh_file
        self.table_mesh_file = table_mesh_file
        self.cfg = cfg
        self.object_root_dir = '/root/catkin_ws/src/config/descriptions/meshes/objects/cuboids'

        self.data_keys = [
            'transformation',
            'contact_obj_frame_right', 'contact_obj_frame_left',
            'contact_world_frame_right', 'contact_world_frame_left',
            'contact_world_frame_2_right', 'contact_world_frame_2_left',
            'object_pointcloud_colors'
        ]

        self.good_camera_euler = [1.0513555,  -0.02236318, -1.62958927]

    def reset_pcd(self, data):
        self.pcd = open3d.geometry.PointCloud()
        self.pcd.points = open3d.utility.Vector3dVector(data['object_pointcloud'])
        self.pcd.estimate_normals(search_param=open3d.geometry.KDTreeSearchParamHybrid(
            radius=0.1, max_nn=30))
        self.kdtree = open3d.geometry.KDTreeFlann(self.pcd)
        self.obj_pointcloud = data['object_pointcloud']

    def vis_palms(self, data, name='grasp', goal='transformation',
                  goal_number=1, palm_number=1,
                  world=False, corr=False, centered=False,
                  ori_rep='quat', full_path=False):
        """Visualize the palms with the object start and goal on the table in the world frame

        Args:
            data (Npz): Data file from numpy post-processed
            name (str, optional): primitive type. Defaults to 'grasp'.
            goal (str, optional): [description]. Defaults to 'transformation'.
            goal_number (int, optional): How many goal locations to view. Defaults to 1.
            palm_number (int, optional): How many palms to view. Defaults to 1.
            world (bool, optional): Whether to use the world frame palm pose or the object frame. Defaults to False.
            corr (bool, optional): Whether to use a corrected palm pose based on the pointcloud. Defaults to False.

        Returns:
            scene
        """
        if ori_rep not in ['quat', 'two_pos']:
            raise ValueError('Unrecognized orientation representation')
        if full_path:
            full_mesh_fname = str(data['mesh_file'])
        else:
            full_mesh_fname = os.path.join(
                self.object_root_dir,
                str(data['mesh_file']))
        obj_mesh = trimesh.load_mesh(full_mesh_fname)
        table_mesh = trimesh.load_mesh(self.table_mesh_file)
        obj_pointcloud = data['object_pointcloud']
        obj_centroid = np.mean(obj_pointcloud, axis=0)

        self.reset_pcd(data)

        if centered:
            offset = obj_centroid
        else:
            offset = np.array([0, 0, 0])

        obj_pos_world, obj_ori_world = data['start_vis'][:3], data['start_vis'][3:]
        h_trans = self.make_h_trans(
            obj_pos_world,
            obj_ori_world
        )

        obj_mesh.apply_transform(h_trans)

        # get the data out of Npz and into a regular dictionary
        viz_data = {}
        viz_data['start_vis'] = data['start_vis']

        # make sure all the data looks like a numpy array
        for key in self.data_keys:
            try:
                if len(data[key].shape) == 1:
                    viz_data[key] = np.expand_dims(data[key], axis=0)
                else:
                    viz_data[key] = data[key]
            except KeyError:
                viz_data[key] = None

        data = viz_data
        goal_obj_meshes = []
        for i in range(goal_number):
            goal_obj_mesh = trimesh.load_mesh(full_mesh_fname)
            if goal == 'transformation':
                T_mat = util.matrix_from_pose(util.list2pose_stamped(data['transformation'][i, :]))
                T_start = util.matrix_from_pose(util.list2pose_stamped(data['start_vis']))
                T_goal = np.matmul(T_mat, T_start)
                goal_obj_pose_world = util.pose_stamped2list(util.pose_from_matrix(T_goal))
                goal_obj_pos_world = goal_obj_pose_world[:3]
                goal_obj_ori_world = goal_obj_pose_world[3:]
            else:
                goal_obj_pos_world = data['goal'][:3]
                goal_obj_ori_world = data['goal'][3:]
            goal_h_trans = self.make_h_trans(
                goal_obj_pos_world,
                goal_obj_ori_world)
            goal_obj_mesh.apply_transform(goal_h_trans)
            goal_obj_meshes.append(goal_obj_mesh)

        r_palm_meshes = []
        l_palm_meshes = []

        if name == 'grasp':
            for j in range(palm_number):
                r_palm_mesh = trimesh.load_mesh(self.palm_mesh_file)
                l_palm_mesh = trimesh.load_mesh(self.palm_mesh_file)

                # use the object frame pose
                if not world:
                    tip_contact_r_obj = util.list2pose_stamped(data['contact_obj_frame_right'][j, :])
                    tip_contact_l_obj = util.list2pose_stamped(data['contact_obj_frame_left'][j, :])

                    tip_contact_r = util.convert_reference_frame(
                        pose_source=tip_contact_r_obj,
                        pose_frame_target=util.unit_pose(),
                        pose_frame_source=util.list2pose_stamped(data['start_vis']))

                    tip_contact_l = util.convert_reference_frame(
                        pose_source=tip_contact_l_obj,
                        pose_frame_target=util.unit_pose(),
                        pose_frame_source=util.list2pose_stamped(data['start_vis']))
                # use the world frame pose
                else:
                    if ori_rep == 'quat':
                        tip_contact_r, tip_contact_l = self.tip_contact_from_quat(
                            data, j, offset, corr
                        )
                    elif ori_rep == 'two_pos':
                        tip_contact_r, tip_contact_l = self.tip_contact_from_two_pos(
                            data, j, offset, corr
                        )

                wrist_contact_r = util.convert_reference_frame(
                    pose_source=util.list2pose_stamped(self.cfg.TIP_TO_WRIST_TF),
                    pose_frame_target=util.unit_pose(),
                    pose_frame_source=tip_contact_r)

                wrist_contact_l = util.convert_reference_frame(
                    pose_source=util.list2pose_stamped(self.cfg.TIP_TO_WRIST_TF),
                    pose_frame_target=util.unit_pose(),
                    pose_frame_source=tip_contact_l)

                wrist_contact_r_list = util.pose_stamped2list(wrist_contact_r)
                wrist_contact_l_list = util.pose_stamped2list(wrist_contact_l)

                palm_pos_world_r = wrist_contact_r_list[:3]
                palm_ori_world_r = wrist_contact_r_list[3:]
                h_trans = self.make_h_trans(palm_pos_world_r, palm_ori_world_r)
                r_palm_mesh.apply_transform(h_trans)

                palm_pos_world_l = wrist_contact_l_list[:3]
                palm_ori_world_l = wrist_contact_l_list[3:]
                h_trans = self.make_h_trans(palm_pos_world_l, palm_ori_world_l)
                l_palm_mesh.apply_transform(h_trans)

                r_palm_meshes.append(r_palm_mesh)
                l_palm_meshes.append(l_palm_mesh)
            scene_list = [obj_mesh, table_mesh] + r_palm_meshes + l_palm_meshes + goal_obj_meshes
            scene = trimesh.Scene(scene_list)

        box_count = 0
        for key in scene.geometry.keys():
            if 'mpalms_all_coarse' in key:
                scene.geometry[key].visual.face_colors = [100, 100, 0, 30]
            if 'realsense_box_experiments' in key or 'cuboid' in key or 'cylinder' in key:
                if box_count == 0:
                    scene.geometry[key].visual.face_colors = [250, 200, 200, 150]
                    box_count += 1
                else:
                    scene.geometry[key].visual.face_colors = [200, 200, 250, 150]

        scene.geometry['table_top.stl'].visual.face_colors = [200, 200, 200, 250]
        scene.set_camera(angles=self.good_camera_euler, center=data['start_vis'][:3], distance=0.8)
        return scene

    def vis_palms_pcd(self, data, name='grasp', goal='transformation',
                      goal_number=1, palm_number=1,
                      world=False, corr=False, centered=False,
                      ori_rep='quat', full_path=False, show_mask=False):
        """Visualize the palms with the object start and goal on the table in the world frame

        Args:
            data (Npz): Data file from numpy post-processed
            name (str, optional): primitive type. Defaults to 'grasp'.
            goal (str, optional): [description]. Defaults to 'transformation'.
            goal_number (int, optional): How many goal locations to view. Defaults to 1.
            palm_number (int, optional): How many palms to view. Defaults to 1.
            world (bool, optional): Whether to use the world frame palm pose or the object frame. Defaults to False.
            corr (bool, optional): Whether to use a corrected palm pose based on the pointcloud. Defaults to False.

        Returns:
            scene
        """
        if ori_rep not in ['quat', 'two_pos']:
            raise ValueError('Unrecognized orientation representation')
        # if full_path:
        #     full_mesh_fname = str(data['mesh_file'])
        # else:
        #     full_mesh_fname = os.path.join(
        #         self.object_root_dir,
        #         str(data['mesh_file']))
        # obj_mesh = trimesh.load_mesh(full_mesh_fname)
        table_mesh = trimesh.load_mesh(self.table_mesh_file)
        table_transform = np.eye(4)
        table_transform[:-1, :-1] = common.euler2rot([0.0, 0.0, np.pi/2])
        table_mesh.apply_transform(table_transform)
        obj_pointcloud = data['start']
        try:
            obj_pointcloud_mask = obj_pointcloud[np.where(data['object_mask'])[0], :]
            # obj_pointcloud_mask = data['object_mask']
        except:
            print('could not get object mask')
            pass

        # obj_pointcloud = data['object_pointcloud']
        obj_centroid = np.mean(obj_pointcloud, axis=0)
        obj_pcd = trimesh.PointCloud(obj_pointcloud)
        if 'object_pointcloud_colors' in data.keys():
            print('loading pointcloud color data')
            obj_pcd.colors = data['object_pointcloud_colors']
        else:
            # obj_pcd.colors = [255, 0, 0, 30]
            obj_pcd.colors = [255, 0, 0, 255]

        self.reset_pcd(data)

        if centered:
            offset = obj_centroid
        else:
            offset = np.array([0, 0, 0])

        # obj_pos_world, obj_ori_world = data['start_vis'][:3], data['start_vis'][3:]
        # h_trans = self.make_h_trans(
        #     obj_pos_world,
        #     obj_ori_world
        # )

        # obj_mesh.apply_transform(h_trans)

        # get the data out of Npz and into a regular dictionary
        viz_data = {}
        # viz_data['start_vis'] = data['start_vis']

        # make sure all the data looks like a numpy array
        for key in self.data_keys:
            try:
                if len(data[key].shape) == 1:
                    viz_data[key] = np.expand_dims(data[key], axis=0)
                else:
                    viz_data[key] = data[key]
            except KeyError:
                viz_data[key] = None

        data = viz_data

        goal_obj_pcds = []
        for i in range(goal_number):
            # goal_obj_mesh = trimesh.load_mesh(full_mesh_fname)
            goal_obj_pcd = trimesh.PointCloud(obj_pointcloud)
            T_mat = util.matrix_from_pose(util.list2pose_stamped(data['transformation'][i, :]))
            goal_obj_pcd.apply_transform(T_mat)
            # if 'object_pointcloud_colors' in data.keys():
            #     print('loading pointcloud color data')
            #     goal_obj_pcd.colors = data['object_pointcloud_colors']
            # else:              
            goal_obj_pcd.colors = [0, 0, 255, 255]
            goal_obj_pcds.append(goal_obj_pcd)

        r_palm_meshes = []
        l_palm_meshes = []

        if name == 'grasp':
            for j in range(palm_number):
                r_palm_mesh = trimesh.load_mesh(self.palm_mesh_file)
                l_palm_mesh = trimesh.load_mesh(self.palm_mesh_file)

                # use the object frame pose
                if not world:
                    raise ValueError('Only accepting world frame palm poses with pointclouds')
                    # tip_contact_r_obj = util.list2pose_stamped(data['contact_obj_frame_right'][j, :])
                    # tip_contact_l_obj = util.list2pose_stamped(data['contact_obj_frame_left'][j, :])

                    # tip_contact_r = util.convert_reference_frame(
                    #     pose_source=tip_contact_r_obj,
                    #     pose_frame_target=util.unit_pose(),
                    #     pose_frame_source=util.list2pose_stamped(data['start_vis']))

                    # tip_contact_l = util.convert_reference_frame(
                    #     pose_source=tip_contact_l_obj,
                    #     pose_frame_target=util.unit_pose(),
                    #     pose_frame_source=util.list2pose_stamped(data['start_vis']))
                # use the world frame pose
                else:
                    if ori_rep == 'quat':
                        tip_contact_r, tip_contact_l = self.tip_contact_from_quat(
                            data, j, offset, corr
                        )
                    elif ori_rep == 'two_pos':
                        tip_contact_r, tip_contact_l = self.tip_contact_from_two_pos(
                            data, j, offset, corr
                        )

                wrist_contact_r = util.convert_reference_frame(
                    pose_source=util.list2pose_stamped(self.cfg.TIP_TO_WRIST_TF),
                    pose_frame_target=util.unit_pose(),
                    pose_frame_source=tip_contact_r)

                wrist_contact_l = util.convert_reference_frame(
                    pose_source=util.list2pose_stamped(self.cfg.TIP_TO_WRIST_TF),
                    pose_frame_target=util.unit_pose(),
                    pose_frame_source=tip_contact_l)

                wrist_contact_r_list = util.pose_stamped2list(wrist_contact_r)
                wrist_contact_l_list = util.pose_stamped2list(wrist_contact_l)

                palm_pos_world_r = wrist_contact_r_list[:3]
                palm_ori_world_r = wrist_contact_r_list[3:]
                h_trans = self.make_h_trans(palm_pos_world_r, palm_ori_world_r)
                r_palm_mesh.apply_transform(h_trans)

                palm_pos_world_l = wrist_contact_l_list[:3]
                palm_ori_world_l = wrist_contact_l_list[3:]
                h_trans = self.make_h_trans(palm_pos_world_l, palm_ori_world_l)
                l_palm_mesh.apply_transform(h_trans)

                r_palm_meshes.append(r_palm_mesh)
                l_palm_meshes.append(l_palm_mesh)
            scene_list = [obj_pcd, table_mesh] + r_palm_meshes + l_palm_meshes + goal_obj_pcds
            if show_mask:
                mask_pcd = trimesh.PointCloud(obj_pointcloud_mask)
                mask_pcd.colors = [0, 255, 0, 255]
                scene_list.append(mask_pcd)
            scene = trimesh.Scene(scene_list)

        box_count = 0
        for key in scene.geometry.keys():
            if 'mpalms_all_coarse' in key:
                scene.geometry[key].visual.face_colors = [0, 153, 153, 200]
            # if 'realsense_box_experiments' in key or 'cuboid' in key or 'cylinder' in key:
            #     if box_count == 0:
            #         scene.geometry[key].visual.face_colors = [250, 200, 200, 150]
            #         box_count += 1
            #     else:
            #         scene.geometry[key].visual.face_colors = [200, 200, 250, 150]

        scene.geometry['table_top.stl'].visual.face_colors = [200, 200, 200, 250]
        scene.set_camera(angles=self.good_camera_euler, center=obj_centroid, distance=0.8)
        return scene

    def tip_contact_from_quat(self, data, ind, offset, corr):
        tip_pos_r = data['contact_world_frame_right'][ind, :3] + offset
        tip_pos_l = data['contact_world_frame_left'][ind, :3] + offset
        # correct the positions based on the pointcloud

        # check if we are pulling or grasping based on whether L and R are the same
        grasping = False if (tip_pos_r == tip_pos_l).all() else True

        if corr:
            if grasping:
                tip_pos = {}
                tip_pos['right'], tip_pos['left'] = tip_pos_r, tip_pos_l
                tip_pos_corr = correct_grasp_pos(tip_pos, self.obj_pointcloud)
                tip_pos_r, tip_pos_l = tip_pos_corr['right'], tip_pos_corr['left']
            else:
                tip_pos_r = correct_palm_pos_single(
                    data['contact_world_frame_right'][ind], 
                    self.obj_pointcloud)[:3]
                tip_pos_l = tip_pos_r
        tip_contact_r = util.list2pose_stamped(
            tip_pos_r.tolist() + data['contact_world_frame_right'][ind, 3:].tolist()
        )
        tip_contact_l = util.list2pose_stamped(
            tip_pos_l.tolist() + data['contact_world_frame_left'][ind, 3:].tolist()
        )
        return tip_contact_r, tip_contact_l

    def tip_contact_from_two_pos(self, data, ind, offset, corr):
        tip_contact_r_world = np.asarray(data['contact_world_frame_right'][ind, :]) + offset
        tip_contact_l_world = np.asarray(data['contact_world_frame_left'][ind, :]) + offset
        tip_contact_r2_world = np.asarray(data['contact_world_frame_2_right'][ind, :]) + offset
        tip_contact_l2_world = np.asarray(data['contact_world_frame_2_left'][ind, :]) + offset

        # get palm y vector
        nearest_pt_ind_r = self.kdtree.search_knn_vector_3d(tip_contact_r_world, 1)[1][0]
        nearest_pt_ind_l = self.kdtree.search_knn_vector_3d(tip_contact_l_world, 1)[1][0]

        normal_y_r = self.pcd.normals[nearest_pt_ind_r]
        normal_y_l = self.pcd.normals[nearest_pt_ind_l]

        # get palm z vector
        palm_z_r = (tip_contact_r2_world - tip_contact_r_world)/np.linalg.norm((tip_contact_r2_world - tip_contact_r_world))
        palm_z_l = (tip_contact_l2_world - tip_contact_l_world)/np.linalg.norm((tip_contact_l2_world - tip_contact_l_world))

        tip_contact_r2_world = util.project_point2plane(
            tip_contact_r2_world + palm_z_r,
            normal_y_r,
            [tip_contact_r_world])[0]
        tip_contact_l2_world = util.project_point2plane(
            tip_contact_l_world + palm_z_l,
            normal_y_l,
            [tip_contact_l_world])[0]

        palm_z_r = (tip_contact_r2_world - tip_contact_r_world)/np.linalg.norm((tip_contact_r2_world - tip_contact_r_world))
        palm_z_l = (tip_contact_l2_world - tip_contact_l_world)/np.linalg.norm((tip_contact_l2_world - tip_contact_l_world))

        if np.dot(normal_y_l, palm_z_l) > 0:
            normal_y_l = -normal_y_l

        x_r, y_r, z_r = np.cross(normal_y_r, palm_z_r), normal_y_r, palm_z_r
        x_l, y_l, z_l = np.cross(normal_y_l, palm_z_l), normal_y_l, palm_z_l

        tip_contact_r = util.pose_from_vectors(
            x_r, y_r, z_r, tip_contact_r_world
        )
        tip_contact_l = util.pose_from_vectors(
            x_l, y_l, z_l, tip_contact_l_world
        )

        return tip_contact_r, tip_contact_l

    def make_spheres(self):
        point_r = trimesh.creation.icosphere(3, radius=0.005, color=[1, 0, 0, 0.9])
        point_l = trimesh.creation.icosphere(3, radius=0.005, color=[0, 0, 1, 0.9])

        point_r_2 = trimesh.creation.icosphere(3, radius=0.005, color=[1, 0, 0, 0.9])
        point_l_2 = trimesh.creation.icosphere(3, radius=0.005, color=[0, 0, 1, 0.9])
        return point_r, point_l, point_r_2, point_l_2

    def make_h_trans(self, pos, ori):
        ori_mat = common.quat2rot(ori)
        h_trans = np.eye(4)
        h_trans[:3, :3] = ori_mat
        h_trans[:-1, -1] = pos
        return h_trans


class PCDVis(object):
    def __init__(self):
        self.setup_plotly()

    def setup_plotly(self):
        self.red_marker = {
            'size' : 1.0,
            'color' : 'red',                # set color to an array/list of desired values
            'colorscale' : 'Viridis',   # choose a colorscale
            'opacity' : 0.5
        }
        self.blue_marker = {
            'size' : 3.0,
            'color' : 'blue',                # set color to an array/list of desired values
            'colorscale' : 'Viridis',   # choose a colorscale
            'opacity' : 1.0
        }

        self.black_square = {
            'size' : 5.0,
            'color' : 'black',                # set color to an array/list of desired values
            'colorscale' : 'Viridis',   # choose a colorscale
            'opacity' : 0.8,
            'symbol': 'square'
        }

        self.blue_square = {
            'size' : 5.0,
            'color' : 'blue',                # set color to an array/list of desired values
            'colorscale' : 'Viridis',   # choose a colorscale
            'opacity' : 0.8,
            'symbol': 'square'
        }

        self.black_tri = {
            'size' : 5.0,
            'color' : 'black',                # set color to an array/list of desired values
            'colorscale' : 'Viridis',   # choose a colorscale
            'opacity' : 0.8,
            'symbol': 'triange'
        }

        self.blue_tri = {
            'size' : 5.0,
            'color' : 'blue',                # set color to an array/list of desired values
            'colorscale' : 'Viridis',   # choose a colorscale
            'opacity' : 0.8,
            'symbol': 'triange'
        }

        self.black_marker = {
            'size' : 1.0,
            'color' : 'black',                # set color to an array/list of desired values
            'colorscale' : 'Viridis',   # choose a colorscale
            'opacity' : 0.8
        }

        self.black_marker_big = {
            'size' : 3.0,
            'color' : 'black',                # set color to an array/list of desired values
            'colorscale' : 'Viridis',   # choose a colorscale
            'opacity' : 0.8
        }

        self.gray_marker = {
            'size': 0.8,
            'color': 'gray',                # set color to an array/list of desired values
            'colorscale': 'Viridis',   # choose a colorscale
            'opacity': 0.5
        }

    def plot_pointcloud(self, data, downsampled=False,
                        importance=False, make_fig=True):
        if downsampled:
            pcd_start = data['start']
            pcd_start_masked = data['start'][np.where(data['object_mask_down'])[0], :]
        else:
            pcd_start = data['object_pointcloud']
            pcd_start_masked = data['object_pointcloud'][np.where(data['object_mask'])[0], :]
        if importance:
            pcd_start = data['start']
            pcd_start_masked = data['start'][np.where(data['object_mask_down'])[0], :]

            impt_inds = data['sort_idx'][0, :10]
            pcd_start_impt = pcd_start[impt_inds, :]

            start_pcd_data_impt = {
                'type': 'scatter3d',
                'x': pcd_start_impt[:, 0],
                'y': pcd_start_impt[:, 1],
                'z': pcd_start_impt[:, 2],
                'mode': 'markers',
                'marker': self.black_marker_big
            }

        start_pcd_data = {
            'type': 'scatter3d',
            'x': pcd_start[:, 0],
            'y': pcd_start[:, 1],
            'z': pcd_start[:, 2],
            'mode': 'markers',
            'marker': self.red_marker
        }

        start_mask_data = {
            'type': 'scatter3d',
            'x': pcd_start_masked[:, 0],
            'y': pcd_start_masked[:, 1],
            'z': pcd_start_masked[:, 2],
            'mode': 'markers',
            'marker': self.blue_marker
        }

        plane_data = {
            'type': 'mesh3d',
            'x': [-1, 1, 1, -1],
            'y': [-1, -1, 1, 1],
            'z': [0, 0, 0, 0],
            'color': 'gray',
            'opacity': 0.5,
            'delaunayaxis': 'z'
        }
        fig_data = []
        fig_data.append(plane_data)
        fig_data.append(start_pcd_data)
        fig_data.append(start_mask_data)
        if importance:
            fig_data.append(start_pcd_data_impt)

        fig = None
        if make_fig:
            fig = go.Figure(data=fig_data)
            camera = {
                'up': {'x': 0, 'y': 0,'z': 1},
                'center': {'x': 0.45, 'y': 0, 'z': 0.0},
                'eye': {'x': -1.0, 'y': 0.0, 'z': 0.01}
            }
            scene = {
                'xaxis': {'nticks': 10, 'range': [-0.1, 0.9]},
                'yaxis': {'nticks': 16, 'range': [-0.5, 0.5]},
                'zaxis': {'nticks': 8, 'range': [-0.01, 0.99]}
            }
            width = 700
            margin = {'r': 20, 'l': 10, 'b': 10, 't': 10}
            fig.update_layout(
                scene=scene,
                scene_camera=camera,
                width=width,
                margin=margin
            )
        return fig, fig_data

    def plot_pointcloud_palm_positions(self, data, downsampled=False,
                                       corr=False, centered=False,
                                       make_fig=True):
        contact_points_r = np.vstack(
            (data['contact_world_frame_right'][:3], data['contact_world_frame_2_right'][:3])
        )
        contact_points_l = np.vstack(
            (data['contact_world_frame_left'][:3], data['contact_world_frame_2_left'][:3])
        )

        if centered:
            contact_points_r = contact_points_r + np.mean(data['object_pointcloud'], axis=0)
            contact_points_l = contact_points_l + np.mean(data['object_pointcloud'], axis=0)

        if corr:
            pass

        contact_data_r = {
            'type': 'scatter3d',
            'x': contact_points_r[:, 0],
            'y': contact_points_r[:, 1],
            'z': contact_points_r[:, 2],
            'mode': 'markers',
            'marker': self.black_square
        }
        contact_data_l = {
            'type': 'scatter3d',
            'x': contact_points_l[:, 0],
            'y': contact_points_l[:, 1],
            'z': contact_points_l[:, 2],
            'mode': 'markers',
            'marker': self.blue_square
        }
        plane_data = {
            'type': 'mesh3d',
            'x': [-1, 1, 1, -1],
            'y': [-1, -1, 1, 1],
            'z': [0, 0, 0, 0],
            'color': 'gray',
            'opacity': 0.5,
            'delaunayaxis': 'z'
        }
        fig_data = self.plot_pointcloud(data, make_fig=False)[1]
        fig_data.append(contact_data_r)
        fig_data.append(contact_data_l)
        fig = None
        if make_fig:
            fig = go.Figure(data=fig_data)
            camera = {
                'up': {'x': 0, 'y': 0,'z': 1},
                'center': {'x': 0.45, 'y': 0, 'z': 0.0},
                'eye': {'x': -1.0, 'y': 0.0, 'z': 0.01}
            }
            scene = {
                'xaxis': {'nticks': 10, 'range': [-0.1, 0.9]},
                'yaxis': {'nticks': 16, 'range': [-0.5, 0.5]},
                'zaxis': {'nticks': 8, 'range': [-0.01, 0.99]}
            }
            width = 700
            margin = {'r': 20, 'l': 10, 'b': 10, 't': 10}
            fig.update_layout(
                scene=scene,
                scene_camera=camera,
                width=width,
                margin=margin
            )
        return fig, fig_data

    def plot_registration_result(self, source, source_transformed, target, make_fig=True):
        source_original_data = {
            'type': 'scatter3d',
            'x': source[:, 0],
            'y': source[:, 1],
            'z': source[:, 2],
            'mode': 'markers',
            'marker': self.blue_marker
        }

        source_transformed_data = {
            'type': 'scatter3d',
            'x': source_transformed[:, 0],
            'y': source_transformed[:, 1],
            'z': source_transformed[:, 2],
            'mode': 'markers',
            'marker': self.red_marker
        }

        target_data = {
            'type': 'scatter3d',
            'x': target[:, 0],
            'y': target[:, 1],
            'z': target[:, 2],
            'mode': 'markers',
            'marker': self.gray_marker
        }

        plane_data = {
            'type': 'mesh3d',
            'x': [-1, 1, 1, -1],
            'y': [-1, -1, 1, 1],
            'z': [0, 0, 0, 0],
            'color': 'gray',
            'opacity': 0.5,
            'delaunayaxis': 'z'
        }
        fig_data = []
        fig_data.append(source_original_data)
        fig_data.append(source_transformed_data)
        fig_data.append(target_data)

        fig = None
        if make_fig:
            fig = go.Figure(data=fig_data)
        #     camera = {
        #         'up': {'x': 0, 'y': 0,'z': 1},
        #         'center': {'x': 0.45, 'y': 0, 'z': 0.0},
        #         'eye': {'x': -1.0, 'y': 0.0, 'z': 0.01}
        #     }
            scene = {
                'xaxis': {'nticks': 10, 'range': [-0.1, 0.9]},
                'yaxis': {'nticks': 16, 'range': [-0.5, 0.5]},
                'zaxis': {'nticks': 8, 'range': [-0.01, 0.99]}
            }
            width = 700
            margin = {'r': 20, 'l': 10, 'b': 10, 't': 10}
            fig.update_layout(
                scene=scene,
                width=width,
                margin=margin
            )
        return fig

