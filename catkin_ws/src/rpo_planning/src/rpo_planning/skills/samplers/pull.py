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
from sampler_utils import PubSubSamplerBase
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


class PullSamplerBasic(object):
    def __init__(self):
        # self.x_bounds = [0.1, 0.5]
        self.x_bounds = [0.2, 0.4]
        # self.y_bounds = [-0.4, 0.4]
        self.y_bounds = [-0.2, 0.2]
        self.theta_bounds = [-np.pi, np.pi]
        # self.theta_bounds = [-2*np.pi/3, 2*np.pi/3]

        self.rand_pull_yaw = lambda: (3*np.pi/4)*np.random.random_sample() + np.pi/4

        self.sample_timeout = 10.0
        self.sample_limit = 100

    def get_model_path(self):
        return 'uniform'        

    def get_transformation(self, state=None, final_trans_to_go=None):
        if final_trans_to_go is not None:
            return final_trans_to_go
        x_pos, y_pos = 0, 0
        if state is not None:
            pos = np.mean(state, axis=0)
            x_pos, y_pos = pos[0], pos[1]
        x = np.random.random() * (max(self.x_bounds) - min(self.x_bounds)) + min(self.x_bounds)
        y = np.random.random() * (max(self.y_bounds) - min(self.y_bounds)) + min(self.y_bounds)
        theta = np.random.random() * (max(self.theta_bounds) - min(self.theta_bounds)) + min(self.theta_bounds)

        trans_to_origin = np.mean(state, axis=0)

        # translate the source to the origin
        T_0 = np.eye(4)
        # T_0[:-1, -1] = -trans_to_origin
        T_0[0, -1] = -trans_to_origin[0]
        T_0[1, -1] = -trans_to_origin[1]

        # apply pure rotation in the world frame, based on prior knowledge that
        # grasping tends to pitch forward/backward
        T_1 = np.eye(4)
        T_1[:-1, :-1] = common.euler2rot([0.0, 0.0, theta])

        # translate in [x, y] back away from origin
        T_2 = np.eye(4)
        T_2[0, -1] = trans_to_origin[0]
        T_2[1, -1] = trans_to_origin[1]

        translate = np.eye(4)
        translate[:2, -1] = np.array([x-x_pos, y-y_pos])

        # compose transformations in correct order
        # transform = np.matmul(T_2, np.matmul(T_1, T_0))
        transform = np.matmul(translate, np.matmul(T_2, np.matmul(T_1, T_0)))

#         print(np.rad2deg(theta))
        # rot = common.euler2rot([0, 0, theta])
        # transform = np.eye(4)
        # transform[:-1, :-1] = rot
        # transform[:2, -1] = np.array([x-x_pos, y-y_pos])
        return transform

    def get_palms(self, state, state_full=None):
        # check if full pointcloud available, if not use sparse pointcloud
        pcd_pts = state if state_full is None else state_full

        # compute pointcloud normals
        pcd = open3d.geometry.PointCloud()
        pcd.points = open3d.utility.Vector3dVector(pcd_pts)
        pcd.estimate_normals()

        pt_samples = []
        dot_samples = []

        # search for a point on the pointcloud with normal pointing up
        # and which is above the center of mass of the object
        sample_i = 0
        sampling_start = time.time()
        while True:
            sample_i += 1
            t = time.time() - sampling_start

            pt_ind = np.random.randint(pcd_pts.shape[0])
            pt_sampled = pcd_pts[pt_ind, :]

            above_com = pt_sampled[2] > np.mean(pcd_pts, axis=0)[2]
            if not above_com:
                continue

            normal_sampled = np.asarray(pcd.normals)[pt_ind, :]

            dot_x = np.abs(np.dot(normal_sampled, [1, 0, 0]))
            dot_y = np.abs(np.dot(normal_sampled, [0, 1, 0]))
            dot_samples.append([dot_x, dot_y])
            pt_samples.append(pt_sampled)

            parallel_z = np.abs(dot_x) < 0.01 and np.abs(dot_y) < 0.01

            if parallel_z:
                break

            if t > self.sample_timeout or sample_i > self.sample_limit:
                dots = np.asarray(dot_samples)
                pts = np.asarray(pt_samples)

                # sort by dot_x
                x_sort_inds = np.argsort(dots[:, 0])
                dot_y_x_sorted = dots[:, 1][x_sort_inds]
                pts_x_sorted = pts[:, :][x_sort_inds]

                # sort those by dot_y
                y_sort_inds = np.argsort(dot_y_x_sorted)
                pts_both_sorted = pts_x_sorted[:, :][y_sort_inds]

                # pick this point
                pt_sampled = pts_both_sorted[0, :]
                break

            # print(dot_x, dot_y, above_com)
            time.sleep(0.01)
        # once this is found turn it into a world frame pose by sampling an orientation
        rand_pull_yaw = self.rand_pull_yaw()
        tip_ori = common.euler2quat([np.pi/2, 0, rand_pull_yaw])
        ori_list = tip_ori.tolist()

        # and converting the known vectors into a pose
        point_list = pt_sampled.tolist()

        world_pose_list = np.asarray(point_list + ori_list)
        return world_pose_list
        # return None

    def sample(self, state=None, state_full=None, final_trans_to_go=None, *args, **kwargs):
        prediction = {}
        prediction['transformation'] = self.get_transformation(state, final_trans_to_go)
        if final_trans_to_go is None:
            prediction['transformation'][2, -1] = 0.0
        prediction['palms'] = self.get_palms(state)
        prediction['mask'] = np.zeros(state.shape)
        return prediction


class PullSamplerVAEPubSub(PubSubSamplerBase):
    def __init__(self, obs_dir, pred_dir, sampler_prefix='pull_vae_', pointnet=False):
        super(PullSamplerVAEPubSub, self).__init__(obs_dir, pred_dir, sampler_prefix)
        self.pointnet = pointnet

        self.x_bounds = [0.1, 0.5]
        self.y_bounds = [-0.4, 0.4]
        self.theta_bounds = [-np.pi, np.pi]

        self.rand_pull_yaw = lambda: (np.pi/2)*np.random.random_sample() + np.pi/2

        self.sample_timeout = 5.0
        self.sample_limit = 100

    def get_palms(self, state, state_full=None):
        # check if full pointcloud available, if not use sparse pointcloud
        if state_full is None:
            pcd_pts = state
        else:
            pcd_pts = state_full

        # compute pointcloud normals
        pcd = open3d.geometry.PointCloud()
        pcd.points = open3d.utility.Vector3dVector(pcd_pts)
        pcd.estimate_normals()

        pt_samples = []
        dot_samples = []

        # search for a point on the pointcloud with normal pointing up
        # and which is above the center of mass of the object
        sample_i = 0
        sampling_start = time.time()
        while True:
            sample_i += 1
            t = time.time() - sampling_start

            pt_ind = np.random.randint(pcd_pts.shape[0])
            pt_sampled = pcd_pts[pt_ind, :]

            above_com = pt_sampled[2] > np.mean(pcd_pts, axis=0)[2]
            if not above_com:
                continue

            normal_sampled = np.asarray(pcd.normals)[pt_ind, :]

            dot_x = np.abs(np.dot(normal_sampled, [1, 0, 0]))
            dot_y = np.abs(np.dot(normal_sampled, [0, 1, 0]))
            dot_samples.append([dot_x, dot_y])
            pt_samples.append(pt_sampled)

            parallel_z = np.abs(dot_x) < 0.01 and np.abs(dot_y) < 0.01

            if parallel_z:
                break

            if t > self.sample_timeout or sample_i > self.sample_limit:
                dots = np.asarray(dot_samples)
                pts = np.asarray(pt_samples)

                # sort by dot_x
                x_sort_inds = np.argsort(dots[:, 0])
                dot_y_x_sorted = dots[:, 1][x_sort_inds]
                pts_x_sorted = pts[:, :][x_sort_inds]

                # sort those by dot_y
                y_sort_inds = np.argsort(dot_y_x_sorted)
                pts_both_sorted = pts_x_sorted[:, :][y_sort_inds]

                # pick this point
                pt_sampled = pts_both_sorted[0, :]
                break

            # print(dot_x, dot_y, above_com)
            time.sleep(0.01)
        # once this is found turn it into a world frame pose by sampling an orientation
        rand_pull_yaw = self.rand_pull_yaw()
        tip_ori = common.euler2quat([np.pi/2, 0, rand_pull_yaw])
        ori_list = tip_ori.tolist()

        # and converting the known vectors into a pose
        point_list = pt_sampled.tolist()

        world_pose_list = np.asarray(point_list + ori_list)
        return world_pose_list

    def sample(self, state=None, state_full=None, final_trans_to_go=None):
        pointcloud_pts = state[:100]        
        prediction = self.filesystem_pub_sub(state)

        # unpack from NN prediction
        ind = np.random.randint(prediction['trans_predictions'].shape[0])
        ind_contact = np.random.randint(20)
        # ind_contact = np.random.randint(2)
        # ind_contact = np.random.randint(40)
        pred_trans_pose = prediction['trans_predictions'][ind, :]
        pred_trans_pos = pred_trans_pose[:3]
        pred_trans_ori = pred_trans_pose[3:]/np.linalg.norm(pred_trans_pose[3:])
        pred_trans_pose = pred_trans_pos.tolist() + pred_trans_ori.tolist()
        pred_trans = np.eye(4)
        pred_trans[:-1, :-1] = common.quat2rot(pred_trans_ori)
        pred_trans[:-1, -1] = pred_trans_pos

        mask = prediction['mask_predictions'][ind, :]
        top_inds = np.argsort(mask)[::-1]
        pred_mask = np.zeros((mask.shape[0]), dtype=bool)
        pred_mask[top_inds[:15]] = True

        try:
            if self.pointnet:
                contact_r = prediction['palm_predictions'][ind, :7]
                contact_l = prediction['palm_predictions'][ind, 7:]
            else:
                contact_r = prediction['palm_predictions'][ind, ind_contact, :7]
                contact_l = prediction['palm_predictions'][ind, ind_contact, 7:]
        except IndexError as e:
            print(e)
            print('index error in pull sampling')
            embed()

        contact_r[:3] += np.mean(pointcloud_pts, axis=0)
        contact_l[:3] += np.mean(pointcloud_pts, axis=0)

        prediction = {}
        if final_trans_to_go is None:
            # force no dz
            prediction['transformation'] = pred_trans
            prediction['transformation'][2, -1] = 0.0
        else:
            prediction['transformation'] = final_trans_to_go
        # prediction['transformation'][2, -1] = 0.0
        prediction['palms'] = contact_r
        # prediction['palms'] = self.get_palms(state, state_full)
        prediction['mask'] = pred_mask
        return prediction