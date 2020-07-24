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
from pointcloud_planning_utils import PointCloudNode, PointCloudCollisionChecker
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


class PointCloudTree(object):
    def __init__(self, start_pcd, trans_des, skeleton, skills,
                 start_pcd_full=None, target=None, motion_planning=True,
                 only_rot=True, target_surfaces=None,
                 visualize=False, obj_id=None, start_pose=None,
                 collision_pcds=None):
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

        self.collision_pcds = collision_pcds
        self.pcd_collision_checker = PointCloudCollisionChecker(self.collision_pcds)

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

                        if self.visualize:
                            sample_pose = util.transform_pose(self.start_pose, util.pose_from_matrix(sample.transformation_so_far))
                            sample_pose_np = util.pose_stamped2np(sample_pose)
                            p.resetBasePositionAndOrientation(self.object_id, sample_pose_np[:3], sample_pose_np[3:])

                        # check if this satisfies the constraints of the next skill
                        valid = self.skills[self.skeleton[i+1]].satisfies_preconditions(sample)
                        
                        # perform 2D pointcloud collision checking, for other objects
                        # valid = valid and self.pcd_collision_checker.check_2d(sample.pointcloud)

                        # check if this is a valid transition (motion planning)
                        if valid:
                            # if self.visualize:
                            #     sample_pose = util.transform_pose(self.start_pose, util.pose_from_matrix(sample.transformation_so_far))
                            #     sample_pose_np = util.pose_stamped2np(sample_pose)
                            #     p.resetBasePositionAndOrientation(self.object_id, sample_pose_np[:3], sample_pose_np[3:])
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
                    sample, index = self.sample_final(i, skill)
                    if self.visualize:
                        sample_pose = util.transform_pose(self.start_pose, util.pose_from_matrix(sample.transformation_so_far))
                        sample_pose_np = util.pose_stamped2np(sample_pose)
                        p.resetBasePositionAndOrientation(self.object_id, sample_pose_np[:3], sample_pose_np[3:])

                    if not self.skills[skill].valid_transformation(sample):
                        # pop sample that this came from
                        self.buffers[i].pop(index)
                        continue

                    # still check motion planning for final step
                    print('final mp')
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
