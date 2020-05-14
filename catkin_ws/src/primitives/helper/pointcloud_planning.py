import os, sys
sys.path.append('/root/catkin_ws/src/primitives/')
import pickle
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import axes3d
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import trimesh
import networkx
import open3d


import copy
import time
import argparse
import numpy as np
# from multiprocessing import Process, Pipe, Queue
import pickle
# import rospy
import copy
import signal
# import open3d
from IPython import embed

from yacs.config import CfgNode as CN
# from closed_loop_experiments import get_cfg_defaults
from closed_loop_experiments_cfg import get_cfg_defaults

# from airobot import Robot
# from airobot.utils import pb_util
# from airobot.sensor.camera.rgbdcam_pybullet import RGBDCameraPybullet
from airobot.utils import common
# import pybullet as p

from helper import util2 as util
# from macro_actions import ClosedLoopMacroActions, YumiGelslimPybulet
# from closed_loop_eval import SingleArmPrimitives, DualArmPrimitives

sys.path.append('/root/training/')

import os
import argparse
import time
import numpy as np
import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
from data_loader import DataLoader
from model import VAE, GoalVAE
from util import to_var, save_state, load_net_state, load_seed, load_opt_state

import scipy.signal as signal

from sklearn.mixture import GaussianMixture
from sklearn.manifold import TSNE

from helper import registration as reg

sys.path.append('/root/training/gat/')
# from model_predictions import *
from models_vae import GoalVAE, GeomVAE, JointVAE


class PointCloudNode(object):
    def __init__(self):
        self.parent = None
        self.pointcloud = None
        self.transformation = None
        self.transformation_to_go = None
        self.palms = None
        
    def set_pointcloud(self, pcd):
        self.pointcloud = pcd
        
    def set_trans_to_go(self, trans):
        self.transformation_to_go = trans
        
    def init_state(self, state, transformation):
        pcd_homog = np.ones((state.pointcloud.shape[0], 4))
        pcd_homog[:, :-1] = state.pointcloud
        self.pointcloud = np.matmul(transformation, pcd_homog.T).T[:, :-1]
        self.transformation = transformation
        self.transformation_to_go = np.matmul(state.transformation_to_go, np.linalg.inv(transformation))

    def init_palms(self, palms):
        self.palms = palms


class PointCloudTree(object):
    def __init__(self, start_pcd, trans_des, skeleton, skills):
        self.skeleton = skeleton
        self.skills = skills
        self.goal_threshold = None

        self.buffers = {}
        for i in range(len(skeleton)):
            self.buffers[i+1] = []
        
        self.start_node = PointCloudNode()
        self.start_node.set_pointcloud(start_pcd)
        self.start_node.set_trans_to_go(trans_des)              
        
        self.buffers[0] = [self.start_node]
        
    def plan(self):
        done = False
        start_time = time.time()
        while not done:
            for i, skill in enumerate(self.skeleton):
                if i < len(self.skeleton) - 1:
                    k = 0
                    while True:
                        k += 1
                        if k > 1000:
                            valid = False
                            break                    
                        sample, index = self.sample_next(i, skill)
                
                        # check if this is a valid transition (motion planning)
                        valid = self.skills[skill].feasible(sample)

                        # check if this satisfies the constraints of the next skill
                        valid = valid and self.skills[self.skeleton[i+1]].satisfies_preconditions(sample)
                        if valid:
                            sample.parent = (i, index)
                            self.buffers[i+1].append(sample)
                            print('saved, yaw: ' + str(np.rad2deg(common.rot2euler(sample.transformation[:-1, :-1])[-1])))
                            break
                else:
                    # sample is the proposed end state, which has the path encoded
                    # via all its parents
                    sample, index = self.sample_next(i, skill)
                    if sample is not None:
                        sample.parent = (i, index)
                        reached_goal = self.reached_goal(sample)
                        if reached_goal:
                            done = True
                            self.buffers['final'] = sample
                            break
                    else:
                        pass
            if time.time() - start_time > 60.0:
                print('Timed out')
                return None
        
        # extract plan
        plan = self.extract_plan()
        return plan
    
    def sample_next(self, i, skill):
        sample, index = None, None
        if i == 0:
            # sample from first skill if starting at beginning
            sample = self.skills[skill].sample(self.start_node)
            index = 0
        else:
            # sample from the buffers we have
            if len(self.buffers[i]) > 0:
                index = np.random.randint(len(self.buffers[i]))
                state = self.buffers[i][index]
                sample = self.skills[skill].sample(state) 
        return sample, index
    
    def reached_goal(self, sample):
        T = np.eye(4)
#         plan = self.extract_plan()
#         for node in plan:
#             T = np.matmul(node.transformation, T)
        T_pose = np.asarray(util.pose_stamped2list(util.pose_from_matrix(T)))
        T_goal = np.asarray(util.pose_stamped2list(util.pose_from_matrix(self.start_node.transformation_to_go)))
#         reached_goal = reach_pose_goal(T_pose, T_goal, pos_tol=0.5, ori_tol=0.05)
#         print(reached_goal, eye_norm)
#         return reached_goal[0]
        eye_norm = np.linalg.norm(sample.transformation_to_go[:-1, :-1] - np.eye(3))        
        return eye_norm < 1.0
    
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

class GraspSamplerVAE(object):
    def __init__(self, model, default_target, latent_dim=256):
        self.dev = torch.device('cuda')
        self.model = model.eval().to(self.dev)
        self.latent_dim = latent_dim
        self.kd_idx = torch.from_numpy(np.arange(100))
        self.default_target = default_target

    def get_transformation(self, state, mask, target=None):
        source = state[np.where(mask)[0], :]
        source_obj = state
        if target is None:
            target = self.default_target

        init_trans_fwd = reg.init_grasp_trans(source, fwd=True)
        init_trans_bwd = reg.init_grasp_trans(source, fwd=False)

        init_trans = init_trans_fwd
        transform = copy.deepcopy(reg.full_registration_np(source, target, init_trans))
        source_obj_trans = reg.apply_transformation_np(source_obj, transform)

#         if np.where(source_obj_trans[:, 2] < 0.005)[0].shape[0] > 100:
        if np.mean(source_obj_trans, axis=0)[2] < 0.005:
            init_trans = init_trans_bwd
            transform = copy.deepcopy(reg.full_registration_np(source, target, init_trans))
            source_obj_trans = reg.apply_transformation_np(source_obj, transform)
        
        return transform
    
    def sample(self, state, target=None):
        state = state[:100]
        state_np = state
        state_mean = np.mean(state, axis=0)
        state_normalized = (state - state_mean)
        state_mean = np.tile(state_mean, (state.shape[0], 1))
        
        state_full = np.concatenate([state_normalized, state_mean], axis=1)
        
        state = torch.from_numpy(state_full)[None, :, :].float().to(self.dev)
        kd_idx = self.kd_idx[None, :].long().to(self.dev)
        
        z = torch.randn(1, self.latent_dim).to(self.dev)
        recon_mu, ex_wt = self.model.decode(z, state)
        output_r, output_l, pred_mask, pred_trans = recon_mu
        
        output_r, output_l = output_r.detach().cpu().numpy(), output_l.detach().cpu().numpy()
        output_joint = np.concatenate([output_r, output_l], axis=2)
        ex_wt = ex_wt.detach().cpu().numpy().squeeze()
        sort_idx = np.argsort(ex_wt)[None, :]
        pred_mask = pred_mask.detach().cpu().numpy().squeeze()
        
        palm_repeat = []
        for i in range(output_joint.shape[0]):
            for j in range(output_joint.shape[1]):
                j = sort_idx[i, j]
                pred_joints = output_joint[i, j]
                palm_repeat.append(pred_joints.tolist())
        
        top_inds = np.argsort(pred_mask)[::-1]
        pred_mask = np.zeros((pred_mask.shape[0]), dtype=bool)
        pred_mask[top_inds[:15]] = True        
        
        prediction = {}
        prediction['palms'] = np.asarray(palm_repeat).squeeze()
        prediction['mask'] = pred_mask
        prediction['transformation'] = self.get_transformation(state_np, pred_mask, target)
        return prediction
    
class PullSamplerBasic(object):
    def __init__(self):
        self.x_bounds = [0.1, 0.5]
        self.y_bounds = [-0.4, 0.4]
        self.theta_bounds = [-np.pi, np.pi]
        
    def get_transformation(self, state=None):
        x_pos, y_pos = 0, 0
        if state is not None:
            pos = np.mean(state, axis=0)
            x_pos, y_pos = pos[0], pos[1]
        x = np.random.random() * (max(self.x_bounds) - min(self.x_bounds)) + min(self.x_bounds)
        y = np.random.random() * (max(self.y_bounds) - min(self.y_bounds)) + min(self.y_bounds)
        theta = np.random.random() * (max(self.theta_bounds) - min(self.theta_bounds)) + min(self.theta_bounds)
#         print(np.rad2deg(theta))
        rot = common.euler2rot([0, 0, theta])
        transform = np.eye(4)
        transform[:-1, :-1] = rot
        transform[:2, -1] = np.array([x-x_pos, y-y_pos])
        return transform
    
    def sample(self, state=None, *args):
        return self.get_transformation(state)


class PrimitiveSkill(object):
    def __init__(self, sampler):
        """Base class for primitive skill

        Args:
            sampler (function): sampling function that generates new
                potential state to add to the plan
        """
        self.sampler = sampler
        self.table_x_min, self.table_x_max = 0.1, 0.5
        self.table_y_min, self.table_y_max = -0.3, 0.3
        
    def satisfies_preconditions(self, state):
        raise NotImplementedError

    def sample(self, state):
        raise NotImplementedError
            
    def object_is_on_table(self, state):
        """
        Checks if pointcloud for this state is within the table boundary
        """
        pos = np.mean(state.pointcloud, axis=0)[:2]
        x, y = pos[0], pos[1]
        x_valid = x > self.table_x_min and x < self.table_x_max
        y_valid = y > self.table_y_min and y < self.table_y_max
        return x_valid and y_valid

class GraspSkill(PrimitiveSkill):
    def __init__(self, sampler):
        super(GraspSkill, self).__init__(sampler)
        self.x_min, self.x_max = 0.3, 0.45
        self.y_min, self.y_max = -0.1, 0.1

    def sample(self, state, target_surface=None):
        # NN sampling, point cloud alignment
        prediction = self.sampler.sample(state.pointcloud, target_surface)
        transformation = prediction['transformation']
        new_state = PointCloudNode()
        new_state.init_state(state, transformation)
        new_state.init_palms(prediction['palms'])
        return new_state
    
    def satisfies_preconditions(self, state):
        # test 1: on the table
        valid = self.object_is_on_table(state)
        valid = valid and self.object_in_grasp_region(state)
        return valid
        
    def object_in_grasp_region(self, state):
        pos = np.mean(state.pointcloud, axis=0)[0:2]
        x, y = pos[0], pos[1]
        x_valid = x < self.x_max and x > self.x_min
        y_valid = y < self.y_max and y > self.y_min
        return x_valid and y_valid
    
    def feasible(self, state):
        # run motion planning on the primitive plan
        # TODO
        return True


class PullSkill(PrimitiveSkill):
    def __init__(self, sampler):
        super(PullSkill, self).__init__(sampler)
        
    def sample(self, state, *args):
        transformation = self.sampler.sample(state.pointcloud)
        new_state = PointCloudNode()
        new_state.init_state(state, transformation)
        return new_state
        
    def satisfies_preconditions(self, state):
        # test 1: on the table
        valid = self.object_is_on_table(state)
        return valid
    
    def feasible(self, state):
        return True