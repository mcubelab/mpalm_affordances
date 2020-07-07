import os, sys
import os.path as osp
import pickle
import numpy as np

import trimesh
import open3d
import pcl
import pybullet as p

import copy
import time
from IPython import embed

from yacs.config import CfgNode as CN
from airobot.utils import common

sys.path.append('/root/catkin_ws/src/primitives/')
# from helper import util2 as util
# from helper import registration as reg
import util2 as util
import registration as reg
from closed_loop_experiments_cfg import get_cfg_defaults
from eval_utils.visualization_tools import correct_grasp_pos, project_point2plane
from pointcloud_planning_utils import PointCloudNode
# from planning import grasp_planning_wf

# sys.path.append('/root/training/')

# import os
# import argparse
# import time
# import numpy as np
# import torch
# from torch import nn
# from torch import optim
# from torch.autograd import Variable
# from data_loader import DataLoader
# from model import VAE, GoalVAE
# from util import to_var, save_state, load_net_state, load_seed, load_opt_state

# import scipy.signal as signal

# from sklearn.mixture import GaussianMixture
# from sklearn.manifold import TSNE

# # sys.path.append('/root/training/gat/')
# # from models_vae import GoalVAE, GeomVAE, JointVAE

# sys.path.append('/root/training/gat/')
# # from models_vae import JointVAE
# from joint_model_vae import JointVAE


# class PointCloudNode(object):
#     def __init__(self):
#         self.parent = None
#         self.pointcloud = None
#         self.pointcloud_full = None
#         self.pointcloud_mask = None
#         self.transformation = None
#         self.transformation_to_go = np.eye(4)
#         self.transformation_so_far = None
#         self.palms = None
#         self.palms_corrected = None
#         self.palms_raw = None

#     def set_pointcloud(self, pcd, pcd_full=None, pcd_mask=None):
#         self.pointcloud = pcd
#         self.pointcloud_full = pcd_full
#         self.pointcloud_mask = pcd_mask

#     def set_trans_to_go(self, trans):
#         self.transformation_to_go = trans

#     def init_state(self, state, transformation):
#         # compute the pointcloud based on the previous pointcloud and specified trans
#         pcd_homog = np.ones((state.pointcloud.shape[0], 4))
#         pcd_homog[:, :-1] = state.pointcloud
#         self.pointcloud = np.matmul(transformation, pcd_homog.T).T[:, :-1]

#         # do the same for the full pointcloud, if it's there
#         if state.pointcloud_full is not None:
#             pcd_full_homog = np.ones((state.pointcloud_full.shape[0], 4))
#             pcd_full_homog[:, :-1] = state.pointcloud_full
#             self.pointcloud_full = np.matmul(transformation, pcd_full_homog.T).T[:, :-1]

#         # node's one step transformation
#         self.transformation = transformation

#         # transformation to go based on previous transformation to go
#         self.transformation_to_go = np.matmul(state.transformation_to_go, np.linalg.inv(transformation))

#         # transformation so far, accounting for if this is the first step,
#         # and parent node has no transformation so far
#         if state.transformation is not None:
#             self.transformation_so_far = np.matmul(transformation, state.transformation_so_far)
#         else:
#             self.transformation_so_far = transformation

#     def init_palms(self, palms, correction=False, prev_pointcloud=None):
#         if correction and prev_pointcloud is not None:
#             palms_raw = palms
#             palms_positions = {}
#             palms_positions['right'] = palms_raw[:3]
#             palms_positions['left'] = palms_raw[7:7+3]
#             pcd_pts = prev_pointcloud
#             palms_positions_corr = correct_grasp_pos(palms_positions,
#                                                      pcd_pts)
#             palm_right_corr = np.hstack([
#                 palms_positions_corr['right'],
#                 palms_raw[3:7]])
#             palm_left_corr = np.hstack([
#                 palms_positions_corr['left'],
#                 palms_raw[7+3:]
#             ])
#             self.palms_corrected = np.hstack([palm_right_corr, palm_left_corr])
#             self.palms_raw = palms_raw
#             self.palms = self.palms_corrected
#         else:
#             self.palms = palms
#             self.palms_raw = palms

#         # check if hands got flipped like a dummy by checking y coordinate in world frame
#         if self.palms.shape[0] > 7:
#             if self.palms[1] > self.palms[1+7]:
#                 tmp_l = copy.deepcopy(self.palms[7:])
#                 self.palms[7:] = copy.deepcopy(self.palms[:7])
#                 self.palms[:7] = tmp_l


class PointCloudTree(object):
    def __init__(self, start_pcd, trans_des, skeleton, skills,
                 start_pcd_full=None, target=None, motion_planning=True,
                 only_rot=True, target_surfaces=None,
                 visualize=False, obj_id=None, start_pose=None):
        self.skeleton = skeleton
        self.skills = skills
        self.goal_threshold = None
        self.motion_planning = motion_planning
        self.timeout = 300
        # self.timeout = 180
        self.only_rot = only_rot

        self.pos_thresh = 0.005
        self.ori_thresh = np.deg2rad(15)
        self.eye_thresh = 1.0

        self.buffers = {}
        for i in range(len(skeleton)):
            self.buffers[i+1] = []

        self.start_node = PointCloudNode()
        self.start_node.set_pointcloud(start_pcd, start_pcd_full)
        self.start_node.set_trans_to_go(trans_des)
        self.transformation = np.eye(4)

        if target_surfaces is None:
            self.target_surfaces = [None]*len(skeleton)
        else:
            self.target_surfaces = target_surfaces
        self.buffers[0] = [self.start_node]

        self.visualize = False
        self.object_id = None
        self.start_pose = None
        if visualize and obj_id is not None and start_pose is not None:
            self.visualize = True
            self.object_id = obj_id
            self.start_pose = start_pose

    def plan(self):
        done = False
        start_time = time.time()
        while not done:
            for i, skill in enumerate(self.skeleton):
                if i < len(self.skeleton) - 1:
                    k = 0
                    while True:
                        if time.time() - start_time > self.timeout:
                            print('Timed out')
                            return None
                        k += 1
                        if k > 1000:
                            valid = False
                            break
                        sample, index = self.sample_next(i, skill)

                        # check if this satisfies the constraints of the next skill
                        # print('checking preconditions')
                        valid = self.skills[self.skeleton[i+1]].satisfies_preconditions(sample)
                        # print('preconditions: ' + str(valid))

                        # check if this is a valid transition (motion planning)
                        if valid:
                            if self.visualize:
                                sample_pose = util.transform_pose(self.start_pose, util.pose_from_matrix(sample.transformation_so_far))
                                sample_pose_np = util.pose_stamped2np(sample_pose)
                                p.resetBasePositionAndOrientation(self.object_id, sample_pose_np[:3], sample_pose_np[3:])
                            if self.motion_planning:
                                valid = valid and self.skills[skill].feasible_motion(sample)
                            else:
                                valid = valid and True
                        else:
                            continue

                        if valid:
                            sample.parent = (i, index)
                            self.buffers[i+1].append(sample)
                            break
                else:
                    # sample is the proposed end state, which has the path encoded
                    # via all its parents
                    # sample, index = self.sample_next(i, skill)
                    # print('sampling final')
                    sample, index = self.sample_final(i, skill)
                    if self.visualize:
                        sample_pose = util.transform_pose(self.start_pose, util.pose_from_matrix(sample.transformation_so_far))
                        sample_pose_np = util.pose_stamped2np(sample_pose)
                        p.resetBasePositionAndOrientation(self.object_id, sample_pose_np[:3], sample_pose_np[3:])

                    if not self.skills[skill].valid_transformation(sample):
                        # pop sample that this came from
                        print('popping sample final')
                        self.buffers[i].pop(index)
                        continue

                    # still check motion planning for final step
                    print('final motion planning')
                    if self.motion_planning:
                        valid = self.skills[skill].feasible_motion(sample)
                    else:
                        valid = True

                    if sample is not None and valid:
                        sample.parent = (i, index)
                        reached_goal = self.reached_goal(sample)
                        if reached_goal:
                            done = True
                            self.buffers['final'] = sample
                            break
                    else:
                        pass
            if time.time() - start_time > self.timeout:
                print('Timed out')
                return None

        # extract plan
        plan = self.extract_plan()
        return plan

    def sample_next(self, i, skill):
        sample, index = None, None
        last_step = i == len(self.skeleton) - 1
        if i == 0:
            # sample from first skill if starting at beginning
            sample = self.skills[skill].sample(
                self.start_node,
                target_surface=self.target_surfaces[i],
                final_trans=last_step)
            index = 0
        else:
            # sample from the buffers we have
            if len(self.buffers[i]) > 0:
                index = np.random.randint(len(self.buffers[i]))
                state = self.buffers[i][index]
                sample = self.skills[skill].sample(
                    state,
                    target_surface=self.target_surfaces[i],
                    final_trans=last_step)
        return sample, index

    def sample_final(self, i, skill):
        sample, index = None, None
        if len(self.buffers[i]) > 0:
            index = np.random.randint(len(self.buffers[i]))
            state = self.buffers[i][index]
            sample = self.skills[skill].sample(
                state,
                final_trans=True)
        return sample, index

    def reached_goal(self, sample):
        T_eye = np.eye(4)
        T_so_far = sample.transformation_so_far
        T_to_go = sample.transformation_to_go

        T_so_far_pose = util.pose_stamped2np(util.pose_from_matrix(T_so_far))
        T_des_pose = util.pose_stamped2np(util.pose_from_matrix(self.start_node.transformation_to_go))

        pos_err, ori_err = util.pose_difference_np(T_so_far_pose, T_des_pose)
        eye_diff_1 = T_to_go[:-1, :-1] - T_eye[:-1, :-1]
        eye_diff_2 = T_to_go - T_eye
        print('pos err: ' + str(pos_err) +
              ' ori err: ' + str(ori_err) +
              ' eye norm 1: ' + str(np.linalg.norm(eye_diff_1)) +
              ' eye norm 2: ' + str(np.linalg.norm(eye_diff_2)))
        if np.isnan(ori_err):
            embed()
        if self.only_rot:
            eye_diff = T_to_go[:-1, :-1] - T_eye[:-1, :-1]
            return ori_err < self.ori_thresh
        else:
            eye_diff = T_to_go - T_eye
            return ori_err < self.ori_thresh and pos_err < self.pos_thresh
        return np.linalg.norm(eye_diff) < self.eye_thresh

    def extract_plan(self):
        node = self.buffers['final']
        parent = node.parent
        plan = []
        plan.append(node)
        while parent is not None:
            node = self.buffers[parent[0]][parent[1]]
            plan.append(node)
            parent = node.parent
        plan.reverse()
        return plan


# class GraspSamplerBasic(object):
#     def __init__(self, default_target):
#         self.default_target = default_target
#         self.sample_timeout = 5.0
#         self.sample_limit = 100
#         self.planes = []

#     def update_default_target(self, target):
#         self.default_target = target

#     def get_transformation(self, state, state_masked,
#                            target=None, final_trans_to_go=None):
#         # source = state[np.where(mask)[0], :]
#         # source_obj = state
#         source = state_masked
#         source_obj = state
#         if target is None:
#             target = self.default_target

#         init_trans_fwd = reg.init_grasp_trans(source, fwd=True)
#         init_trans_bwd = reg.init_grasp_trans(source, fwd=False)

#         if final_trans_to_go is None:
#             init_trans = init_trans_fwd
#         else:
#             init_trans = final_trans_to_go
#         transform = copy.deepcopy(reg.full_registration_np(source, target, init_trans))
#         source_obj_trans = reg.apply_transformation_np(source_obj, transform)

#         if np.mean(source_obj_trans, axis=0)[2] < 0.005:
#             init_trans = init_trans_bwd
#             transform = copy.deepcopy(reg.full_registration_np(source, target, init_trans))
#             source_obj_trans = reg.apply_transformation_np(source_obj, transform)

#         return transform

#     def get_palms(self, state, state_full=None):
#         # check if full pointcloud available, if not use sparse pointcloud
#         pcd_pts = state if state_full is None else state_full

#         # compute pointcloud normals
#         pcd = open3d.geometry.PointCloud()
#         pcd.points = open3d.utility.Vector3dVector(pcd_pts)
#         pcd.estimate_normals()

#         pt_samples = []
#         dot_samples = []
#         normal_samples = []

#         # search for a point on the pointcloud with normal NOT pointing up
#         sample_i = 0
#         sampling_start = time.time()
#         while True:
#             sample_i += 1
#             t = time.time() - sampling_start

#             pt_ind = np.random.randint(pcd_pts.shape[0])
#             pt_sampled = pcd_pts[pt_ind, :]

#             normal_sampled = np.asarray(pcd.normals)[pt_ind, :]

#             dot_x = np.abs(np.dot(normal_sampled, [1, 0, 0]))
#             dot_y = np.abs(np.dot(normal_sampled, [0, 1, 0]))
#             dot_z = np.abs(np.dot(normal_sampled, [0, 0, 1]))

#             dot_samples.append(dot_z)
#             pt_samples.append(pt_sampled)
#             normal_samples.append(normal_sampled)

#             orthogonal_z = np.abs(dot_z) < 0.01

#             if orthogonal_z:
#                 break

#             if t > self.sample_timeout or sample_i > self.sample_limit:
#                 dots = np.asarray(dot_samples)
#                 pts = np.asarray(pt_samples)
#                 normals = np.asarray(normal_sampled)

#                 # sort by dot_x
#                 z_sort_inds = np.argsort(dots)
#                 dots_z_sorted = dots[z_sort_inds]
#                 pts_z_sorted = pts[z_sort_inds, :]
#                 normal_z_sorted = normals[z_sort_inds, :]

#                 # pick this point
#                 pt_sampled = pts_z_sorted[0, :]
#                 normal_sampled = normal_z_sorted[0, :]
#                 break

#             # print(dot_x, dot_y, above_com)
#             time.sleep(0.01)

#         # get both normal directions
#         normal_vec_1 = normal_sampled
#         normal_vec_2 = -normal_sampled

#         # get endpoint based on normals
#         endpoint_1 = pt_sampled + normal_vec_1
#         endpoint_2 = pt_sampled + normal_vec_2

#         # get points interpolated between endpoints and sampled point
#         points_along_1_0 = np.linspace(pt_sampled, endpoint_1, 2000)
#         points_along_1_1 = np.linspace(endpoint_1, pt_sampled, 2000)
#         points_along_2_0 = np.linspace(pt_sampled, endpoint_2, 2000)
#         points_along_2_1 = np.linspace(endpoint_2, pt_sampled, 2000)

#         one_points = [
#             points_along_1_0,
#             points_along_1_1
#         ]
#         two_points = [
#             points_along_2_0,
#             points_along_2_1
#         ]
#         one_norms = [
#             np.linalg.norm(points_along_1_0[0] - points_along_1_0[-1]),
#             np.linalg.norm(points_along_1_1[0] - points_along_1_1[-1])
#         ]
#         two_norms = [
#             np.linalg.norm(points_along_2_0[0] - points_along_2_0[-1]),
#             np.linalg.norm(points_along_2_1[0] - points_along_2_1[-1])
#         ]
#         points_along_1 = one_points[np.argmax(one_norms)]
#         points_along_2 = two_points[np.argmax(two_norms)]

#         points = {}
#         points['1'] = points_along_1
#         points['2'] = points_along_2

#         dists = {}
#         dists['1'] = []
#         dists['2'] = []

#         inds = {}
#         inds['1'] = []
#         inds['2'] = []

#         # go through all points to find the one with the smallest distance to the pointcloud
#         kdtree = open3d.geometry.KDTreeFlann(pcd)
#         for key in points.keys():
#             for i in range(points[key].shape[0]):
#                 pos = points[key][i, :]
#                 nearest_pt_ind = kdtree.search_knn_vector_3d(pos, 1)[1][0]

#                 dist = np.asarray(pcd.points)[nearest_pt_ind] - pos

#                 inds[key].append((i, nearest_pt_ind))
#                 dists[key].append(dist.dot(dist))

#         opposite_pt_candidates = []
#         for key in points.keys():
#             min_ind = np.argmin(dists[key])
#             inds_sorted = np.argsort(dists[key])
#             # for i, min_ind in enumerate(inds_sorted[:10]):
#             min_ind = inds_sorted[-1]
#             min_dist = dists[key][min_ind]
#             min_point_ind = inds[key][min_ind][0]
#             min_point_pcd_ind = inds[key][min_ind][1]
#             nearest_pt_world = points[key][min_point_ind]
#             nearest_pt_pcd_world = np.asarray(pcd.points)[min_point_pcd_ind]
#             # opposite_pt_candidates.append((nearest_pt_world, min_dist))
#             opposite_pt_candidates.append((nearest_pt_pcd_world, min_dist))

#         # pick which one based on smaller distance
#         dist_vals = [opposite_pt_candidates[0][1], opposite_pt_candidates[1][1]]
#         opp_pt_sampled = opposite_pt_candidates[np.argmin(dist_vals)][0]

#         # guess which one should be right and left based on y values
#         both_pts_sampled = [pt_sampled, opp_pt_sampled]
#         y_vals = [pt_sampled[1], opp_pt_sampled[1]]
#         r_pt_ind, l_pt_ind = np.argmin(y_vals), np.argmax(y_vals)

#         tip_contact_r_world = both_pts_sampled[r_pt_ind]
#         tip_contact_l_world = both_pts_sampled[l_pt_ind]

#         # sample a random 3D vector to get second points
#         rand_dir = np.random.random(3) * 2.0 - 1.0
#         second_pt_dir = rand_dir/np.linalg.norm(rand_dir)
#         tip_contact_r2_world = tip_contact_r_world + second_pt_dir
#         tip_contact_l2_world = tip_contact_l_world + second_pt_dir

#         # get palm y vector
#         nearest_pt_ind_r = kdtree.search_knn_vector_3d(tip_contact_r_world, 1)[1][0]
#         nearest_pt_ind_l = kdtree.search_knn_vector_3d(tip_contact_l_world, 1)[1][0]

#         normal_y_r = pcd.normals[nearest_pt_ind_r]
#         normal_y_l = pcd.normals[nearest_pt_ind_l]

#         # get palm z vector
#         # palm_z_r = (tip_contact_r2_world - tip_contact_r_world)/np.linalg.norm((tip_contact_r2_world - tip_contact_r_world))
#         # palm_z_l = (tip_contact_l2_world - tip_contact_l_world)/np.linalg.norm((tip_contact_l2_world - tip_contact_l_world))

#         tip_contact_r2_world = project_point2plane(
#             tip_contact_r2_world,
#             normal_y_r,
#             [tip_contact_r_world])[0]
#         tip_contact_l2_world = project_point2plane(
#             tip_contact_l2_world,
#             normal_y_l,
#             [tip_contact_l_world])[0]

#         palm_z_r = (tip_contact_r2_world - tip_contact_r_world)/np.linalg.norm((tip_contact_r2_world - tip_contact_r_world))
#         palm_z_l = (tip_contact_l2_world - tip_contact_l_world)/np.linalg.norm((tip_contact_l2_world - tip_contact_l_world))

#         if np.dot(normal_y_l, palm_z_l) > 0:
#             normal_y_l = -normal_y_l

#         x_r, y_r, z_r = np.cross(normal_y_r, palm_z_r), normal_y_r, palm_z_r
#         x_l, y_l, z_l = np.cross(normal_y_l, palm_z_l), normal_y_l, palm_z_l

#         com = np.mean(pcd_pts, axis=0)
#         com_r_vec = tip_contact_r_world - com
#         if np.dot(com_r_vec, y_r) < 0.0:
#             tmp = tip_contact_r_world
#             tip_contact_r_world = tip_contact_l_world
#             tip_contact_l_world = tmp

#         tip_contact_r = util.pose_from_vectors(
#             x_r, y_r, z_r, tip_contact_r_world
#         )
#         # tip_contact_l = util.pose_from_vectors(
#         #     x_l, y_l, z_l, tip_contact_l_world
#         # )
#         tip_contact_l = util.pose_from_vectors(
#             -x_r, -y_r, z_r, tip_contact_l_world
#         )

#         tip_contact_r_np = util.pose_stamped2np(tip_contact_r)
#         tip_contact_l_np = util.pose_stamped2np(tip_contact_l)
#         # embed()

#         # print(pt_sampled, opp_pt_sampled)
#         # from eval_utils.visualization_tools import PalmVis
#         # from multistep_planning_eval_cfg import get_cfg_defaults
#         # cfg = get_cfg_defaults()
#         # # prep visualization tools
#         # palm_mesh_file = osp.join(os.environ['CODE_BASE'],
#         #                             cfg.PALM_MESH_FILE)
#         # table_mesh_file = osp.join(os.environ['CODE_BASE'],
#         #                             cfg.TABLE_MESH_FILE)
#         # viz_palms = PalmVis(palm_mesh_file, table_mesh_file, cfg)
#         # viz_data = {}
#         # viz_data['contact_world_frame_right'] = tip_contact_r_np
#         # viz_data['contact_world_frame_left'] = tip_contact_l_np
#         # viz_data['transformation'] = util.pose_stamped2np(util.unit_pose())
#         # viz_data['object_pointcloud'] = pcd_pts
#         # viz_data['start'] = pcd_pts

#         # scene_pcd = viz_palms.vis_palms_pcd(viz_data, world=True, corr=False, full_path=True, show_mask=False, goal_number=1)
#         # scene_pcd.show()

#         # embed()

#         return np.concatenate([tip_contact_r_np, tip_contact_l_np])

#     def get_pointcloud_planes(self, pointcloud):
#         planes = []

#         original_pointcloud = copy.deepcopy(pointcloud)
#         com_z = np.mean(original_pointcloud, axis=0)[2]

#         for _ in range(5):
#             inliers = self.segment_pointcloud(pointcloud)
#             masked_pts = pointcloud[inliers]
#             pcd = open3d.geometry.PointCloud()
#             pcd.points = open3d.utility.Vector3dVector(masked_pts)
#             pcd.estimate_normals()

#             masked_pts_z_mean = np.mean(masked_pts, axis=0)[2]
#             above_com = masked_pts_z_mean > com_z

#             parallel_z = 0
#             for _ in range(100):
#                 pt_ind = np.random.randint(masked_pts.shape[0])
#                 pt_sampled = masked_pts[pt_ind, :]
#                 normal_sampled = np.asarray(pcd.normals)[pt_ind, :]

#                 dot_x = np.abs(np.dot(normal_sampled, [1, 0, 0]))
#                 dot_y = np.abs(np.dot(normal_sampled, [0, 1, 0]))
#                 if np.abs(dot_x) < 0.01 and np.abs(dot_y) < 0.01:
#                     parallel_z += 1

#             # print(parallel_z)
#             if not (above_com and parallel_z > 30):
#                 # don't consider planes that are above the CoM
#                 plane_dict = {}
#                 plane_dict['mask'] = inliers
#                 plane_dict['points'] = masked_pts
#                 planes.append(plane_dict)

#             # from eval_utils.visualization_tools import PalmVis
#             # from multistep_planning_eval_cfg import get_cfg_defaults
#             # cfg = get_cfg_defaults()
#             # # prep visualization tools
#             # palm_mesh_file = osp.join(os.environ['CODE_BASE'],
#             #                             cfg.PALM_MESH_FILE)
#             # table_mesh_file = osp.join(os.environ['CODE_BASE'],
#             #                             cfg.TABLE_MESH_FILE)
#             # viz_palms = PalmVis(palm_mesh_file, table_mesh_file, cfg)
#             # viz_data = {}
#             # viz_data['contact_world_frame_right'] = util.pose_stamped2np(util.unit_pose())
#             # viz_data['contact_world_frame_left'] = util.pose_stamped2np(util.unit_pose())
#             # viz_data['transformation'] = util.pose_stamped2np(util.unit_pose())
#             # viz_data['object_pointcloud'] = masked_pts
#             # viz_data['start'] = masked_pts

#             # scene_pcd = viz_palms.vis_palms_pcd(viz_data, world=True, corr=False, full_path=True, show_mask=False, goal_number=1)
#             # scene_pcd.show()

#             pointcloud = np.delete(pointcloud, inliers, axis=0)
#         return planes

#     def segment_pointcloud(self, pointcloud):
#         p = pcl.PointCloud(np.asarray(pointcloud, dtype=np.float32))

#         seg = p.make_segmenter_normals(ksearch=50)
#         seg.set_optimize_coefficients(True)
#         seg.set_model_type(pcl.SACMODEL_PLANE)
#         seg.set_normal_distance_weight(0.05)
#         seg.set_method_type(pcl.SAC_RANSAC)
#         seg.set_max_iterations(100)
#         seg.set_distance_threshold(0.005)
#         inliers, _ = seg.segment()

#         # plane_pts = p.to_array()[inliers]
#         # return plane_pts
#         return inliers

#     def sample(self, state=None, state_full=None, target=None, final_trans_to_go=None):
#         pointcloud_pts = state if state_full is None else state_full

#         # get mask, based on plane fitting and other heuristics
#         planes = self.get_pointcloud_planes(pointcloud_pts)
#         # pred_mask = self.segment_pointcloud(pointcloud_pts)
#         # pred_mask = planes[np.random.randint(5)]['mask']
#         pred_points_masked = planes[np.random.randint(len(planes))]['points']

#         prediction = {}
#         prediction['transformation'] = self.get_transformation(
#             pointcloud_pts,
#             pred_points_masked,
#             target,
#             final_trans_to_go)
#         prediction['palms'] = self.get_palms(state, state_full)
#         prediction['mask'] = pred_points_masked
#         return prediction


# class GraspSamplerVAE(object):
#     def __init__(self, model, default_target, latent_dim=256):
#         self.dev = torch.device('cuda')
#         self.model = model.eval().to(self.dev)
#         self.latent_dim = latent_dim
#         self.kd_idx = torch.from_numpy(np.arange(100))
#         self.default_target = default_target

#     def get_transformation(self, state, mask, target=None, final_trans_to_go=None):
#         source = state[np.where(mask)[0], :]
#         source_obj = state
#         if target is None:
#             target = self.default_target

#         init_trans_fwd = reg.init_grasp_trans(source, fwd=True)
#         init_trans_bwd = reg.init_grasp_trans(source, fwd=False)

#         if final_trans_to_go is None:
#             init_trans = init_trans_fwd
#         else:
#             init_trans = final_trans_to_go
#         transform = copy.deepcopy(reg.full_registration_np(source, target, init_trans))
#         source_obj_trans = reg.apply_transformation_np(source_obj, transform)

# #         if np.where(source_obj_trans[:, 2] < 0.005)[0].shape[0] > 100:
#         if np.mean(source_obj_trans, axis=0)[2] < 0.005:
#             init_trans = init_trans_bwd
#             transform = copy.deepcopy(reg.full_registration_np(source, target, init_trans))
#             source_obj_trans = reg.apply_transformation_np(source_obj, transform)

#         return transform

#     def sample(self, state, target=None, final_trans_to_go=None, *args, **kwargs):
#         state = state[:100]
#         state_np = state
#         state_mean = np.mean(state, axis=0)
#         state_normalized = (state - state_mean)
#         state_mean = np.tile(state_mean, (state.shape[0], 1))

#         state_full = np.concatenate([state_normalized, state_mean], axis=1)

#         mask_predictions = []
#         palm_predictions = []
#         xy_predictions = []
#         joint_keypoint = torch.from_numpy(state_full)[None, :, :].float().to(self.dev)
#         for repeat in range(10):
#             palm_repeat = []
#             z = torch.randn(1, self.latent_dim).to(self.dev)
#             recon_mu, ex_wt = self.model.decode(z, joint_keypoint)
#             output_r, output_l, pred_mask, pred_trans = recon_mu
#             mask_predictions.append(pred_mask.detach().cpu().numpy())
#             xy_predictions.append(pred_trans.detach().cpu().numpy())

#             output_r, output_l = output_r.detach().cpu().numpy(), output_l.detach().cpu().numpy()
#             output_joint = np.concatenate([output_r, output_l], axis=2)
#             ex_wt = ex_wt.detach().cpu().numpy().squeeze()
#             # sort_idx = np.argsort(ex_wt, axis=1)[:, ::-1]
#             sort_idx = np.argsort(ex_wt)[None, :]

#             for i in range(output_joint.shape[0]):
#                 for j in range(output_joint.shape[1]):
#                     j = sort_idx[i, j]
#                     pred_info = output_joint[i, j]
#             #         pred_info = obj_frame[i].cpu().numpy()
#                     palm_repeat.append(pred_info.tolist())
#             palm_predictions.append(palm_repeat)
#         palm_predictions = np.asarray(palm_predictions).squeeze()
#         mask_predictions = np.asarray(mask_predictions).squeeze()
#         xy_predictions = np.asarray(xy_predictions).squeeze()

#         mask_ind = np.random.randint(10)
#         palm_ind = np.random.randint(5)
#         pred_mask = mask_predictions[mask_ind]
#         pred_palm = palm_predictions[mask_ind, palm_ind, :]
#         pred_xy = xy_predictions[mask_ind]
#         # print('(x, y) prediction: ', pred_xy)

#         top_inds = np.argsort(pred_mask)[::-1]
#         pred_mask = np.zeros((pred_mask.shape[0]), dtype=bool)
#         pred_mask[top_inds[:15]] = True

#         prediction = {}
#         prediction['palms'] = pred_palm
#         prediction['mask'] = pred_mask
#         pred_transform = self.get_transformation(state_np, pred_mask, target)
#         dxy = np.random.random(2) * 0.2 - 0.1
#         pred_transform[:2, -1] += dxy
#         print('(x, y) prediction: ', dxy)
#         prediction['transformation'] = pred_transform
#         return prediction


# class GraspSamplerVAEPubSub(object):
#     def __init__(self, default_target, obs_dir, pred_dir, pointnet=False):
#         self.default_target = default_target
#         self.obs_dir = obs_dir
#         self.pred_dir = pred_dir
#         self.samples_count = 0
#         self.pointnet = pointnet
#         self.sampler_prefix = 'grasp_vae_'

#     def update_default_target(self, target):
#         self.default_target = target

#     def get_transformation(self, state, mask,
#                            target=None, final_trans_to_go=None,
#                            pp=False):
#         if pp:
#             source = state
#         source = state[np.where(mask)[0], :]
#         source_obj = state
#         inplace = False
#         if target is None:
#             target = self.default_target
#             inplace = True

#         # init_trans_fwd = reg.init_grasp_trans(source, fwd=True)
#         # init_trans_bwd = reg.init_grasp_trans(source, fwd=False)
#         init_trans_fwd = reg.init_grasp_trans(source, fwd=True, target=target, inplace=inplace, pp=pp)
#         init_trans_bwd = reg.init_grasp_trans(source, fwd=False, target=target, inplace=inplace, pp=pp)

#         if not pp:
#             if final_trans_to_go is None:
#                 init_trans = init_trans_fwd
#             else:
#                 init_trans = final_trans_to_go
#             transform = copy.deepcopy(reg.full_registration_np(source, target, init_trans))
#             source_obj_trans = reg.apply_transformation_np(source_obj, transform)

#             # if np.mean(source_obj_trans, axis=0)[2] < 0.005:
#             if np.mean(source_obj_trans, axis=0)[2] < np.mean(target, axis=0)[2] * 1.05:
#                 init_trans = init_trans_bwd
#                 transform = copy.deepcopy(reg.full_registration_np(source, target, init_trans))
#                 source_obj_trans = reg.apply_transformation_np(source_obj, transform)
#         else:
#             transform = init_trans_fwd

#         return transform

#     def sample(self, state, target=None, final_trans_to_go=None, pp=False, *args, **kwargs):
#         self.samples_count += 1
#         # put inputs inside numpy file
#         T_mat = np.eye(4)
#         transformation = np.asarray(util.pose_stamped2list(util.pose_from_matrix(T_mat)), dtype=np.float32)
#         pointcloud_pts = state[:100]

#         # write to known location
#         obs_fname = osp.join(
#             self.obs_dir,
#             self.sampler_prefix + str(self.samples_count) + '.npz')
#         np.savez(
#             obs_fname,
#             pointcloud_pts=pointcloud_pts,
#             transformation=transformation
#         )

#         # wait for return
#         got_file = False
#         pred_fname = osp.join(
#             self.pred_dir,
#             self.sampler_prefix + str(self.samples_count) + '.npz')
#         start = time.time()
#         while True:
#             try:
#                 prediction = np.load(pred_fname)
#                 got_file = True
#             except:
#                 pass
#             if got_file or (time.time() - start > 300):
#                 break
#             time.sleep(0.01)
#         # if not got_file:
#         #     wait = raw_input('waiting for predictions to come back online')
#         #     return
#         os.remove(pred_fname)

#         # unpack from returned file
#         ind = np.random.randint(prediction['mask_predictions'].shape[0])
#         ind_contact = np.random.randint(5)

#         mask = prediction['mask_predictions'][ind, :]
#         top_inds = np.argsort(mask)[::-1]
#         pred_mask = np.zeros((mask.shape[0]), dtype=bool)
#         pred_mask[top_inds[:15]] = True

#         # embed()

#         # fix palm predictions (via CoM)
#         if self.pointnet:
#             contact_r = prediction['palm_predictions'][ind, :7]
#             contact_l = prediction['palm_predictions'][ind, 7:]
#         else:
#             contact_r = prediction['palm_predictions'][ind, ind_contact, :7]
#             contact_l = prediction['palm_predictions'][ind, ind_contact, 7:]

#         contact_r[:3] += np.mean(pointcloud_pts, axis=0)
#         contact_l[:3] += np.mean(pointcloud_pts, axis=0)

#         # put into local prediction
#         prediction_dict = {}
#         prediction_dict['palms'] = np.hstack([contact_r, contact_l])
#         prediction_dict['mask'] = pred_mask
#         prediction_dict['transformation'] = self.get_transformation(
#             pointcloud_pts,
#             pred_mask,
#             target,
#             final_trans_to_go,
#             pp=pp)
#         return prediction_dict


# class GraspSamplerTransVAEPubSub(object):
#     def __init__(self, default_target, obs_dir, pred_dir, pointnet=False):
#         self.default_target = None
#         self.obs_dir = obs_dir
#         self.pred_dir = pred_dir
#         self.samples_count = 0
#         self.pointnet = pointnet
#         self.sampler_prefix = 'grasp_vae_T_'

#     def update_default_target(self, target):
#         pass

#     def sample(self, state, target=None, final_trans_to_go=None, *args, **kwargs):
#         self.samples_count += 1
#         # put inputs inside numpy file
#         T_mat = np.eye(4)
#         transformation = np.asarray(util.pose_stamped2list(util.pose_from_matrix(T_mat)), dtype=np.float32)
#         pointcloud_pts = state[:100]

#         # write to known location
#         obs_fname = osp.join(
#             self.obs_dir,
#             self.sampler_prefix + str(self.samples_count) + '.npz')
#         np.savez(
#             obs_fname,
#             pointcloud_pts=pointcloud_pts,
#             transformation=transformation
#         )

#         # wait for return
#         got_file = False
#         pred_fname = osp.join(
#             self.pred_dir,
#             self.sampler_prefix + str(self.samples_count) + '.npz')
#         start = time.time()
#         while True:
#             try:
#                 prediction = np.load(pred_fname)
#                 got_file = True
#             except:
#                 pass
#             if got_file or (time.time() - start > 300):
#                 break
#             time.sleep(0.01)
#         # if not got_file:
#         #     wait = raw_input('waiting for predictions to come back online')
#         #     return
#         os.remove(pred_fname)

#         # unpack from returned file
#         ind = np.random.randint(prediction['trans_predictions'].shape[0])
#         ind_contact = np.random.randint(5)
#         pred_trans = prediction['trans_predictions'][ind]

#         # embed()

#         # fix palm predictions (via CoM)
#         if self.pointnet:
#             contact_r = prediction['palm_predictions'][ind, :7]
#             contact_l = prediction['palm_predictions'][ind, 7:]
#         else:
#             contact_r = prediction['palm_predictions'][ind, ind_contact, :7]
#             contact_l = prediction['palm_predictions'][ind, ind_contact, 7:]

#         contact_r[:3] += np.mean(pointcloud_pts, axis=0)
#         contact_l[:3] += np.mean(pointcloud_pts, axis=0)

#         # put into local prediction
#         prediction_dict = {}
#         prediction_dict['palms'] = np.hstack([contact_r, contact_l])
#         prediction_dict['mask'] = np.zeros_like(pointcloud_pts)
#         prediction_dict['transformation'] = util.matrix_from_pose(
#             util.list2pose_stamped(pred_trans))
#         return prediction_dict


# class PullSamplerBasic(object):
#     def __init__(self):
#         # self.x_bounds = [0.1, 0.5]
#         self.x_bounds = [0.2, 0.4]
#         # self.y_bounds = [-0.4, 0.4]
#         self.y_bounds = [-0.2, 0.2]
#         self.theta_bounds = [-np.pi, np.pi]
#         # self.theta_bounds = [-2*np.pi/3, 2*np.pi/3]

#         self.rand_pull_yaw = lambda: (3*np.pi/4)*np.random.random_sample() + np.pi/4

#         self.sample_timeout = 10.0
#         self.sample_limit = 100

#     def get_transformation(self, state=None, final_trans_to_go=None):
#         if final_trans_to_go is not None:
#             return final_trans_to_go
#         x_pos, y_pos = 0, 0
#         if state is not None:
#             pos = np.mean(state, axis=0)
#             x_pos, y_pos = pos[0], pos[1]
#         x = np.random.random() * (max(self.x_bounds) - min(self.x_bounds)) + min(self.x_bounds)
#         y = np.random.random() * (max(self.y_bounds) - min(self.y_bounds)) + min(self.y_bounds)
#         theta = np.random.random() * (max(self.theta_bounds) - min(self.theta_bounds)) + min(self.theta_bounds)

#         trans_to_origin = np.mean(state, axis=0)

#         # translate the source to the origin
#         T_0 = np.eye(4)
#         # T_0[:-1, -1] = -trans_to_origin
#         T_0[0, -1] = -trans_to_origin[0]
#         T_0[1, -1] = -trans_to_origin[1]

#         # apply pure rotation in the world frame, based on prior knowledge that
#         # grasping tends to pitch forward/backward
#         T_1 = np.eye(4)
#         T_1[:-1, :-1] = common.euler2rot([0.0, 0.0, theta])

#         # translate in [x, y] back away from origin
#         T_2 = np.eye(4)
#         T_2[0, -1] = trans_to_origin[0]
#         T_2[1, -1] = trans_to_origin[1]

#         translate = np.eye(4)
#         translate[:2, -1] = np.array([x-x_pos, y-y_pos])

#         # compose transformations in correct order
#         # transform = np.matmul(T_2, np.matmul(T_1, T_0))
#         transform = np.matmul(translate, np.matmul(T_2, np.matmul(T_1, T_0)))

# #         print(np.rad2deg(theta))
#         # rot = common.euler2rot([0, 0, theta])
#         # transform = np.eye(4)
#         # transform[:-1, :-1] = rot
#         # transform[:2, -1] = np.array([x-x_pos, y-y_pos])
#         return transform

#     def get_palms(self, state, state_full=None):
#         # check if full pointcloud available, if not use sparse pointcloud
#         pcd_pts = state if state_full is None else state_full

#         # compute pointcloud normals
#         pcd = open3d.geometry.PointCloud()
#         pcd.points = open3d.utility.Vector3dVector(pcd_pts)
#         pcd.estimate_normals()

#         pt_samples = []
#         dot_samples = []

#         # search for a point on the pointcloud with normal pointing up
#         # and which is above the center of mass of the object
#         sample_i = 0
#         sampling_start = time.time()
#         while True:
#             sample_i += 1
#             t = time.time() - sampling_start

#             pt_ind = np.random.randint(pcd_pts.shape[0])
#             pt_sampled = pcd_pts[pt_ind, :]

#             above_com = pt_sampled[2] > np.mean(pcd_pts, axis=0)[2]
#             if not above_com:
#                 continue

#             normal_sampled = np.asarray(pcd.normals)[pt_ind, :]

#             dot_x = np.abs(np.dot(normal_sampled, [1, 0, 0]))
#             dot_y = np.abs(np.dot(normal_sampled, [0, 1, 0]))
#             dot_samples.append([dot_x, dot_y])
#             pt_samples.append(pt_sampled)

#             parallel_z = np.abs(dot_x) < 0.01 and np.abs(dot_y) < 0.01

#             if parallel_z:
#                 break

#             if t > self.sample_timeout or sample_i > self.sample_limit:
#                 dots = np.asarray(dot_samples)
#                 pts = np.asarray(pt_samples)

#                 # sort by dot_x
#                 x_sort_inds = np.argsort(dots[:, 0])
#                 dot_y_x_sorted = dots[:, 1][x_sort_inds]
#                 pts_x_sorted = pts[:, :][x_sort_inds]

#                 # sort those by dot_y
#                 y_sort_inds = np.argsort(dot_y_x_sorted)
#                 pts_both_sorted = pts_x_sorted[:, :][y_sort_inds]

#                 # pick this point
#                 pt_sampled = pts_both_sorted[0, :]
#                 break

#             # print(dot_x, dot_y, above_com)
#             time.sleep(0.01)
#         # once this is found turn it into a world frame pose by sampling an orientation
#         rand_pull_yaw = self.rand_pull_yaw()
#         tip_ori = common.euler2quat([np.pi/2, 0, rand_pull_yaw])
#         ori_list = tip_ori.tolist()

#         # and converting the known vectors into a pose
#         point_list = pt_sampled.tolist()

#         world_pose_list = np.asarray(point_list + ori_list)
#         return world_pose_list
#         # return None

#     def sample(self, state=None, state_full=None, final_trans_to_go=None):
#         prediction = {}
#         prediction['transformation'] = self.get_transformation(state, final_trans_to_go)
#         prediction['palms'] = self.get_palms(state)
#         prediction['mask'] = np.zeros(state.shape)
#         return prediction


# class PullSamplerVAEPubSub(object):
#     def __init__(self, obs_dir, pred_dir, pointnet=False):
#         self.obs_dir = obs_dir
#         self.pred_dir = pred_dir
#         self.samples_count = 0
#         self.pointnet = pointnet
#         self.sampler_prefix = 'pull_vae_'

#         self.x_bounds = [0.1, 0.5]
#         self.y_bounds = [-0.4, 0.4]
#         self.theta_bounds = [-np.pi, np.pi]

#         self.rand_pull_yaw = lambda: (np.pi/2)*np.random.random_sample() + np.pi/2

#         self.sample_timeout = 5.0
#         self.sample_limit = 100

#     def get_palms(self, state, state_full=None):
#         # check if full pointcloud available, if not use sparse pointcloud
#         if state_full is None:
#             pcd_pts = state
#         else:
#             pcd_pts = state_full

#         # compute pointcloud normals
#         pcd = open3d.geometry.PointCloud()
#         pcd.points = open3d.utility.Vector3dVector(pcd_pts)
#         pcd.estimate_normals()

#         pt_samples = []
#         dot_samples = []

#         # search for a point on the pointcloud with normal pointing up
#         # and which is above the center of mass of the object
#         sample_i = 0
#         sampling_start = time.time()
#         while True:
#             sample_i += 1
#             t = time.time() - sampling_start

#             pt_ind = np.random.randint(pcd_pts.shape[0])
#             pt_sampled = pcd_pts[pt_ind, :]

#             above_com = pt_sampled[2] > np.mean(pcd_pts, axis=0)[2]
#             if not above_com:
#                 continue

#             normal_sampled = np.asarray(pcd.normals)[pt_ind, :]

#             dot_x = np.abs(np.dot(normal_sampled, [1, 0, 0]))
#             dot_y = np.abs(np.dot(normal_sampled, [0, 1, 0]))
#             dot_samples.append([dot_x, dot_y])
#             pt_samples.append(pt_sampled)

#             parallel_z = np.abs(dot_x) < 0.01 and np.abs(dot_y) < 0.01

#             if parallel_z:
#                 break

#             if t > self.sample_timeout or sample_i > self.sample_limit:
#                 dots = np.asarray(dot_samples)
#                 pts = np.asarray(pt_samples)

#                 # sort by dot_x
#                 x_sort_inds = np.argsort(dots[:, 0])
#                 dot_y_x_sorted = dots[:, 1][x_sort_inds]
#                 pts_x_sorted = pts[:, :][x_sort_inds]

#                 # sort those by dot_y
#                 y_sort_inds = np.argsort(dot_y_x_sorted)
#                 pts_both_sorted = pts_x_sorted[:, :][y_sort_inds]

#                 # pick this point
#                 pt_sampled = pts_both_sorted[0, :]
#                 break

#             # print(dot_x, dot_y, above_com)
#             time.sleep(0.01)
#         # once this is found turn it into a world frame pose by sampling an orientation
#         rand_pull_yaw = self.rand_pull_yaw()
#         tip_ori = common.euler2quat([np.pi/2, 0, rand_pull_yaw])
#         ori_list = tip_ori.tolist()

#         # and converting the known vectors into a pose
#         point_list = pt_sampled.tolist()

#         world_pose_list = np.asarray(point_list + ori_list)
#         return world_pose_list

#     def sample(self, state=None, state_full=None, final_trans_to_go=None):
#         self.samples_count += 1

#         # put inputs inside npz file for NN to get on the other end
#         T_mat = np.eye(4)
#         transformation = util.pose_stamped2np(util.pose_from_matrix(T_mat))
#         pointcloud_pts = state[:100]

#         obs_fname = osp.join(
#             self.obs_dir,
#             self.sampler_prefix + str(self.samples_count) + '.npz')
#         np.savez(
#             obs_fname,
#             pointcloud_pts=pointcloud_pts,
#             transformation=transformation
#         )

#         # wait for return
#         got_file = False
#         pred_fname = osp.join(
#             self.pred_dir,
#             self.sampler_prefix + str(self.samples_count) + '.npz')
#         start = time.time()
#         while True:
#             try:
#                 prediction = np.load(pred_fname)
#                 got_file = True
#             except:
#                 pass
#             if got_file or (time.time() - start > 300):
#                 break
#             time.sleep(0.01)
#         os.remove(pred_fname)

#         ind = np.random.randint(prediction['trans_predictions'].shape[0])
#         ind_contact = np.random.randint(5)
#         pred_trans_pose = prediction['trans_predictions'][ind, :]
#         pred_trans_pos = pred_trans_pose[:3]
#         pred_trans_ori = pred_trans_pose[3:]/np.linalg.norm(pred_trans_pose[3:])
#         pred_trans_pose = pred_trans_pos.tolist() + pred_trans_ori.tolist()
#         pred_trans = np.eye(4)
#         pred_trans[:-1, :-1] = common.quat2rot(pred_trans_ori)
#         pred_trans[:-1, -1] = pred_trans_pos

#         mask = prediction['mask_predictions'][ind, :]
#         top_inds = np.argsort(mask)[::-1]
#         pred_mask = np.zeros((mask.shape[0]), dtype=bool)
#         pred_mask[top_inds[:15]] = True

#         contact_r = prediction['palm_predictions'][ind, ind_contact, :7]
#         contact_l = prediction['palm_predictions'][ind, ind_contact, 7:]

#         # contact_r[:3] += np.mean(pointcloud_pts, axis=0)
#         # contact_l[:3] += np.mean(pointcloud_pts, axis=0)

#         prediction = {}
#         if final_trans_to_go is None:
#             prediction['transformation'] = pred_trans
#         else:
#             prediction['transformation'] = final_trans_to_go
#         # prediction['palms'] = contact_r
#         prediction['palms'] = self.get_palms(state, state_full)
#         prediction['mask'] = pred_mask
#         return prediction


# class PrimitiveSkill(object):
#     def __init__(self, sampler, robot):
#         """Base class for primitive skill

#         Args:
#             robot (TODO): Interface to the robot (PyBullet, MoveIt, ROS)
#             sampler (function): sampling function that generates new
#                 potential state to add to the plan
#         """
#         self.robot = robot
#         self.sampler = sampler
#         self.table_x_min, self.table_x_max = 0.1, 0.5
#         self.table_y_min, self.table_y_max = -0.3, 0.3

#     def valid_transformation(self, state):
#         raise NotImplementedError

#     def satisfies_preconditions(self, state):
#         raise NotImplementedError

#     def sample(self, state, target_surface=None, final_trans=False):
#         raise NotImplementedError

#     def object_is_on_table(self, state):
#         """
#         Checks if pointcloud for this state is within the table boundary
#         """
#         pos = np.mean(state.pointcloud, axis=0)[:2]
#         x, y = pos[0], pos[1]
#         x_valid = x > self.table_x_min and x < self.table_x_max
#         y_valid = y > self.table_y_min and y < self.table_y_max
#         # return x_valid and y_valid
#         return True


# class GraspSkill(PrimitiveSkill):
#     def __init__(self, sampler, robot, get_plan_func, ignore_mp=False, pp=False):
#         super(GraspSkill, self).__init__(sampler, robot)
#         self.x_min, self.x_max = 0.35, 0.45
#         self.y_min, self.y_max = -0.1, 0.1
#         self.start_joints = [0.9936, -2.1848, -0.9915, 0.8458, 3.7618,  1.5486,  0.1127,
#                             -1.0777, -2.1187, 0.995, 1.002 ,  -3.6834,  1.8132,  2.6405]
#         self.get_plan_func = get_plan_func
#         self.ignore_mp = ignore_mp
#         self.pick_and_place = pp

#     def get_nominal_plan(self, plan_args):
#         # from planning import grasp_planning_wf
#         palm_pose_l_world = plan_args['palm_pose_l_world']
#         palm_pose_r_world = plan_args['palm_pose_r_world']
#         transformation = plan_args['transformation']
#         N = plan_args['N']

#         nominal_plan = self.get_plan_func(
#             palm_pose_l_world=palm_pose_l_world,
#             palm_pose_r_world=palm_pose_r_world,
#             transformation=transformation,
#             N=N
#         )

#         return nominal_plan

#     def valid_transformation(self, state):
#         # TODO: check if too much roll
#         return True

#     def sample(self, state, target_surface=None, final_trans=False):
#         # NN sampling, point cloud alignment
#         if final_trans:
#             prediction = self.sampler.sample(
#                 state=state.pointcloud,
#                 state_full=state.pointcloud_full,
#                 target=target_surface,
#                 final_trans_to_go=state.transformation_to_go)
#         else:
#             prediction = self.sampler.sample(
#                 state=state.pointcloud,
#                 state_full=state.pointcloud_full,
#                 target=target_surface,
#                 pp=self.pick_and_place)
#         transformation = prediction['transformation']
#         new_state = PointCloudNode()
#         new_state.init_state(state, transformation)
#         new_state.init_palms(prediction['palms'],
#                              correction=True,
#                              prev_pointcloud=state.pointcloud_full)
#         return new_state

#     def satisfies_preconditions(self, state):
#         # test 1: on the table
#         valid = self.object_is_on_table(state)

#         # test 2: in front of the robot
#         valid = valid and self.object_in_grasp_region(state)
#         return valid

#     def object_in_grasp_region(self, state):
#         # checks if the CoM is in a nice region in front of the robot
#         pos = np.mean(state.pointcloud, axis=0)[0:2]
#         x, y = pos[0], pos[1]
#         x_valid = x < self.x_max and x > self.x_min
#         y_valid = y < self.y_max and y > self.y_min
#         return x_valid and y_valid

#     def feasible_motion(self, state, start_joints=None, nominal_plan=None):
#         if self.ignore_mp:
#             return True
#         if nominal_plan is None:
#             # construct plan args
#             plan_args = {}
#             plan_args['palm_pose_l_world'] = util.list2pose_stamped(
#                 state.palms[7:].tolist())
#             plan_args['palm_pose_r_world'] = util.list2pose_stamped(
#                 state.palms[:7].tolist()
#             )
#             plan_args['transformation'] = util.pose_from_matrix(state.transformation)
#             plan_args['N'] = 60

#             # get primitive plan
#             nominal_plan = self.get_nominal_plan(plan_args)

#         right_valid = []
#         left_valid = []

#         for subplan_number, subplan_dict in enumerate(nominal_plan):
#             subplan_tip_poses = subplan_dict['palm_poses_world']

#             # setup motion planning request with all the cartesian waypoints
#             tip_right = []
#             tip_left = []

#             # bump y a bit in the palm frame for pre pose, for collision avoidance
#             if subplan_number == 0:
#                 pre_pose_right_init = util.unit_pose()
#                 pre_pose_left_init = util.unit_pose()

#                 pre_pose_right_init.pose.position.y += 0.05
#                 pre_pose_left_init.pose.position.y += 0.05

#                 pre_pose_right = util.transform_pose(
#                     pre_pose_right_init, subplan_tip_poses[0][1])

#                 pre_pose_left = util.transform_pose(
#                     pre_pose_left_init, subplan_tip_poses[0][0])

#                 tip_right.append(pre_pose_right.pose)
#                 tip_left.append(pre_pose_left.pose)

#             for i in range(len(subplan_tip_poses)):
#                 tip_right.append(subplan_tip_poses[i][1].pose)
#                 tip_left.append(subplan_tip_poses[i][0].pose)

#             if start_joints is None:
#                 # l_start = self.robot.get_jpos(arm='left')
#                 # r_start = self.robot.get_jpos(arm='right')
#                 l_start = self.start_joints[7:]
#                 r_start = self.start_joints[:7]
#             else:
#                 l_start = start_joints['left']
#                 r_start = start_joints['right']

#             try:
#                 self.robot.mp_right.plan_waypoints(
#                     tip_right,
#                     force_start=l_start+r_start,
#                     avoid_collisions=False
#                 )
#                 right_valid.append(1)
#             except ValueError as e:
#                 break
#             try:
#                 self.robot.mp_left.plan_waypoints(
#                     tip_left,
#                     force_start=l_start+r_start,
#                     avoid_collisions=False
#                 )
#                 left_valid.append(1)
#             except ValueError as e:
#                 break
#         valid = False
#         if sum(right_valid) == len(nominal_plan) and \
#                 sum(left_valid) == len(nominal_plan):
#             valid = True
#         return valid


# class PullRightSkill(PrimitiveSkill):
#     def __init__(self, sampler, robot, get_plan_func, ignore_mp=False, avoid_collisions=True):
#         super(PullRightSkill, self).__init__(sampler, robot)
#         self.get_plan_func = get_plan_func
#         self.start_joints_r = [0.417, -1.038, -1.45, 0.26, 0.424, 1.586, 2.032]
#         self.start_joint_l = [-0.409, -1.104, 1.401, 0.311, -0.403, 1.304, 1.142]
#         self.unit_n = 100
#         self.ignore_mp = ignore_mp
#         self.avoid_collisions = avoid_collisions

#     def get_nominal_plan(self, plan_args):
#         # from planning import grasp_planning_wf
#         palm_pose_l_world = plan_args['palm_pose_l_world']
#         palm_pose_r_world = plan_args['palm_pose_r_world']
#         transformation = plan_args['transformation']
#         N = plan_args['N']

#         nominal_plan = self.get_plan_func(
#             palm_pose_l_world=palm_pose_l_world,
#             palm_pose_r_world=palm_pose_r_world,
#             transformation=transformation,
#             N=N
#         )

#         return nominal_plan

#     def valid_transformation(self, state):
#         return self.within_se2_margin(state.transformation)

#     def sample(self, state, *args, **kwargs):
#         final_trans = False
#         if 'final_trans' in kwargs.keys():
#             final_trans = kwargs['final_trans']
#         if final_trans:
#             final_trans_to_go = state.transformation_to_go
#         else:
#             final_trans_to_go = None

#         pcd_pts = state.pointcloud
#         pcd_pts_full = None
#         if state.pointcloud_full is not None:
#             pcd_pts_full = state.pointcloud_full

#         prediction = self.sampler.sample(
#             pcd_pts,
#             state_full=pcd_pts_full,
#             final_trans_to_go=final_trans_to_go)
#         new_state = PointCloudNode()
#         new_state.init_state(state, prediction['transformation'])
#         new_state.init_palms(prediction['palms'])
#         return new_state

#     def satisfies_preconditions(self, state):
#         # test 1: on the table
#         valid = self.object_is_on_table(state)
#         return valid

#     def calc_n(self, dx, dy):
#         dist = np.sqrt(dx**2 + dy**2)
#         N = max(2, int(dist*self.unit_n))
#         return N

#     def within_se2_margin(self, transformation):
#         euler = common.rot2euler(transformation[:-1, :-1])
#         # print('euler: ', euler)
#         return np.abs(euler[0]) < np.deg2rad(20) and np.abs(euler[1]) < np.deg2rad(20)

#     def feasible_motion(self, state, start_joints=None, nominal_plan=None):
#         # # check if transformation is within margin of pure SE(2) transformation
#         if not self.within_se2_margin(state.transformation):
#             return False

#         if self.ignore_mp:
#             return True

#         # construct plan args
#         if nominal_plan is None:
#             plan_args = {}
#             # just copying the right to the left, cause it's not being used anyways
#             plan_args['palm_pose_l_world'] = util.list2pose_stamped(
#                 state.palms[:7].tolist())
#             plan_args['palm_pose_r_world'] = util.list2pose_stamped(
#                 state.palms[:7].tolist()
#             )
#             plan_args['transformation'] = util.pose_from_matrix(state.transformation)
#             plan_args['N'] = self.calc_n(state.transformation[0, -1],
#                                          state.transformation[1, -1])

#             # get primitive plan
#             nominal_plan = self.get_nominal_plan(plan_args)

#         subplan_tip_poses = nominal_plan[0]['palm_poses_world']

#         # setup motion planning request with cartesian waypoints
#         tip_right, tip_left = [], []

#         # create an approach waypoint near the object
#         pre_pose_right_init = util.unit_pose()
#         pre_pose_left_init = util.unit_pose()

#         pre_pose_right_init.pose.position.y += 0.05
#         pre_pose_left_init.pose.position.y += 0.05

#         pre_pose_right = util.transform_pose(pre_pose_right_init,
#                                              subplan_tip_poses[0][1])
#         pre_pose_left = util.transform_pose(pre_pose_left_init,
#                                             subplan_tip_poses[0][0])
#         tip_right.append(pre_pose_right.pose)
#         tip_left.append(pre_pose_left.pose)

#         # create all other cartesian waypoints
#         for i in range(len(subplan_tip_poses)):
#             tip_right.append(subplan_tip_poses[i][1].pose)
#             tip_left.append(subplan_tip_poses[i][0].pose)

#         if start_joints is None:
#             r_start = self.start_joints_r
#             l_start = self.start_joint_l
#         else:
#             r_start = start_joints['right']
#             l_start = start_joints['left']

#         # l_start = self.robot.get_jpos(arm='left')

#         # plan cartesian path
#         valid = False
#         try:
#             # self.robot.mp_right.plan_waypoints(
#             #     tip_right,
#             #     force_start=l_start+r_start,
#             #     avoid_collisions=False
#             # )
#             self.mp_func(
#                 tip_right,
#                 tip_left,
#                 force_start=l_start+r_start
#             )
#             valid = True
#         except ValueError:
#             pass

#         return valid

#     def mp_func(self, tip_right, tip_left, force_start):
#         self.robot.mp_right.plan_waypoints(
#             tip_right,
#             force_start=force_start,
#             avoid_collisions=self.avoid_collisions
#         )


# class PullLeftSkill(PullRightSkill):
#     def __init__(self, sampler, robot, get_plan_func, ignore_mp=False, avoid_collisions=True):
#         super(PullLeftSkill, self).__init__(sampler, robot, get_plan_func, ignore_mp, avoid_collisions)

#     def sample(self, state, *args, **kwargs):
#         final_trans = False
#         if 'final_trans' in kwargs.keys():
#             final_trans = kwargs['final_trans']
#         if final_trans:
#             final_trans_to_go = state.transformation_to_go
#         else:
#             final_trans_to_go = None

#         pcd_pts = copy.deepcopy(state.pointcloud)
#         pcd_pts[:, 1] = -pcd_pts[:, 1]
#         pcd_pts_full = None
#         if state.pointcloud_full is not None:
#             pcd_pts_full = copy.deepcopy(state.pointcloud_full)
#             pcd_pts_full[:, 1] = -pcd_pts_full[:, 1]

#         prediction = self.sampler.sample(
#             pcd_pts,
#             state_full=pcd_pts_full,
#             final_trans_to_go=final_trans_to_go)

#         if final_trans:
#             # on last step we have to trust that this is correct
#             new_transformation = prediction['transformation']
#         else:
#             # if in the middle, then flip based on the right pull transform
#             new_transformation = copy.deepcopy(prediction['transformation'])
#             new_transformation[0, 1] *= -1
#             new_transformation[1, 0] *= -1
#             new_transformation[1, -1] *= -1
#         new_palms = util.pose_stamped2np(util.flip_palm_pulling(util.list2pose_stamped(prediction['palms'][:7])))
#         new_palms[1] *= -1

#         new_state = PointCloudNode()
#         new_state.init_state(state, new_transformation)
#         new_state.init_palms(new_palms)
#         return new_state

#     def mp_func(self, tip_right, tip_left, force_start):
#         self.robot.mp_left.plan_waypoints(
#             tip_left,
#             force_start=force_start,
#             avoid_collisions=self.avoid_collisions
#         )