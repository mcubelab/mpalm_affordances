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
from skill_utils import PrimitiveSkill
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


class GraspSkill(PrimitiveSkill):
    def __init__(self, sampler, robot, get_plan_func, ignore_mp=False, pp=False):
        super(GraspSkill, self).__init__(sampler, robot)
        self.x_min, self.x_max = 0.35, 0.45
        self.y_min, self.y_max = -0.1, 0.1
        self.start_joints = [0.9936, -2.1848, -0.9915, 0.8458, 3.7618,  1.5486,  0.1127,
                            -1.0777, -2.1187, 0.995, 1.002 ,  -3.6834,  1.8132,  2.6405]
        self.get_plan_func = get_plan_func
        self.ignore_mp = ignore_mp
        self.pick_and_place = pp

    def get_nominal_plan(self, plan_args):
        # from planning import grasp_planning_wf
        palm_pose_l_world = plan_args['palm_pose_l_world']
        palm_pose_r_world = plan_args['palm_pose_r_world']
        transformation = plan_args['transformation']
        N = plan_args['N']

        nominal_plan = self.get_plan_func(
            palm_pose_l_world=palm_pose_l_world,
            palm_pose_r_world=palm_pose_r_world,
            transformation=transformation,
            N=N
        )

        return nominal_plan

    def valid_transformation(self, state):
        # TODO: check if too much roll
        return True

    def sample(self, state, target_surface=None, final_trans=False):
        # NN sampling, point cloud alignment
        if final_trans:
            prediction = self.sampler.sample(
                state=state.pointcloud,
                state_full=state.pointcloud_full,
                target=target_surface,
                final_trans_to_go=state.transformation_to_go)
        else:
            prediction = self.sampler.sample(
                state=state.pointcloud,
                state_full=state.pointcloud_full,
                target=target_surface,
                pp=self.pick_and_place)
        transformation = prediction['transformation']
        new_state = PointCloudNode()
        new_state.init_state(state, transformation)
        new_state.init_palms(prediction['palms'],
                             correction=True,
                             prev_pointcloud=state.pointcloud_full)
        return new_state

    def satisfies_preconditions(self, state):
        # test 1: on the table
        valid = self.object_is_on_table(state)

        # test 2: in front of the robot
        valid = valid and self.object_in_grasp_region(state)
        return valid

    def object_in_grasp_region(self, state):
        # checks if the CoM is in a nice region in front of the robot
        pos = np.mean(state.pointcloud, axis=0)[0:2]
        x, y = pos[0], pos[1]
        x_valid = x < self.x_max and x > self.x_min
        y_valid = y < self.y_max and y > self.y_min
        return x_valid and y_valid

    def feasible_motion(self, state, start_joints=None, nominal_plan=None):
        if self.ignore_mp:
            return True
        if nominal_plan is None:
            # construct plan args
            plan_args = {}
            plan_args['palm_pose_l_world'] = util.list2pose_stamped(
                state.palms[7:].tolist())
            plan_args['palm_pose_r_world'] = util.list2pose_stamped(
                state.palms[:7].tolist()
            )
            plan_args['transformation'] = util.pose_from_matrix(state.transformation)
            plan_args['N'] = 60

            # get primitive plan
            nominal_plan = self.get_nominal_plan(plan_args)

        right_valid = []
        left_valid = []

        for subplan_number, subplan_dict in enumerate(nominal_plan):
            subplan_tip_poses = subplan_dict['palm_poses_world']

            # setup motion planning request with all the cartesian waypoints
            tip_right = []
            tip_left = []

            # bump y a bit in the palm frame for pre pose, for collision avoidance
            if subplan_number == 0:
                pre_pose_right_init = util.unit_pose()
                pre_pose_left_init = util.unit_pose()

                pre_pose_right_init.pose.position.y += 0.05
                pre_pose_left_init.pose.position.y += 0.05

                pre_pose_right = util.transform_pose(
                    pre_pose_right_init, subplan_tip_poses[0][1])

                pre_pose_left = util.transform_pose(
                    pre_pose_left_init, subplan_tip_poses[0][0])

                tip_right.append(pre_pose_right.pose)
                tip_left.append(pre_pose_left.pose)

            for i in range(len(subplan_tip_poses)):
                tip_right.append(subplan_tip_poses[i][1].pose)
                tip_left.append(subplan_tip_poses[i][0].pose)

            if start_joints is None:
                # l_start = self.robot.get_jpos(arm='left')
                # r_start = self.robot.get_jpos(arm='right')
                l_start = self.start_joints[7:]
                r_start = self.start_joints[:7]
            else:
                l_start = start_joints['left']
                r_start = start_joints['right']

            try:
                self.robot.mp_right.plan_waypoints(
                    tip_right,
                    force_start=l_start+r_start,
                    avoid_collisions=False
                )
                right_valid.append(1)
            except ValueError as e:
                break
            try:
                self.robot.mp_left.plan_waypoints(
                    tip_left,
                    force_start=l_start+r_start,
                    avoid_collisions=False
                )
                left_valid.append(1)
            except ValueError as e:
                break
        valid = False
        if sum(right_valid) == len(nominal_plan) and \
                sum(left_valid) == len(nominal_plan):
            valid = True
        return valid


class PullRightSkill(PrimitiveSkill):
    def __init__(self, sampler, robot, get_plan_func, ignore_mp=False, avoid_collisions=True):
        super(PullRightSkill, self).__init__(sampler, robot)
        self.get_plan_func = get_plan_func
        self.start_joints_r = [0.417, -1.038, -1.45, 0.26, 0.424, 1.586, 2.032]
        self.start_joint_l = [-0.409, -1.104, 1.401, 0.311, -0.403, 1.304, 1.142]
        self.unit_n = 100
        self.ignore_mp = ignore_mp
        self.avoid_collisions = avoid_collisions

    def get_nominal_plan(self, plan_args):
        # from planning import grasp_planning_wf
        palm_pose_l_world = plan_args['palm_pose_l_world']
        palm_pose_r_world = plan_args['palm_pose_r_world']
        transformation = plan_args['transformation']
        N = plan_args['N']

        nominal_plan = self.get_plan_func(
            palm_pose_l_world=palm_pose_l_world,
            palm_pose_r_world=palm_pose_r_world,
            transformation=transformation,
            N=N
        )

        return nominal_plan

    def valid_transformation(self, state):
        return self.within_se2_margin(state.transformation)

    def sample(self, state, *args, **kwargs):
        final_trans = False
        if 'final_trans' in kwargs.keys():
            final_trans = kwargs['final_trans']
        if final_trans:
            final_trans_to_go = state.transformation_to_go
        else:
            final_trans_to_go = None

        pcd_pts = state.pointcloud
        pcd_pts_full = None
        if state.pointcloud_full is not None:
            pcd_pts_full = state.pointcloud_full

        prediction = self.sampler.sample(
            pcd_pts,
            state_full=pcd_pts_full,
            final_trans_to_go=final_trans_to_go)
        new_state = PointCloudNode()
        new_state.init_state(state, prediction['transformation'])
        # new_state.init_palms(prediction['palms'])
        new_state.init_palms(prediction['palms'],
                             correction=True,
                             prev_pointcloud=pcd_pts_full,
                             dual=False)        
        return new_state

    def satisfies_preconditions(self, state):
        # test 1: on the table
        valid = self.object_is_on_table(state)
        return valid

    def calc_n(self, dx, dy):
        dist = np.sqrt(dx**2 + dy**2)
        N = max(2, int(dist*self.unit_n))
        return N

    def within_se2_margin(self, transformation):
        euler = common.rot2euler(transformation[:-1, :-1])
        # print('euler: ', euler)
        return np.abs(euler[0]) < np.deg2rad(20) and np.abs(euler[1]) < np.deg2rad(20)

    def feasible_motion(self, state, start_joints=None, nominal_plan=None):
        # # check if transformation is within margin of pure SE(2) transformation
        if not self.within_se2_margin(state.transformation):
            return False

        if self.ignore_mp:
            return True

        # construct plan args
        if nominal_plan is None:
            plan_args = {}
            # just copying the right to the left, cause it's not being used anyways
            plan_args['palm_pose_l_world'] = util.list2pose_stamped(
                state.palms[:7].tolist())
            plan_args['palm_pose_r_world'] = util.list2pose_stamped(
                state.palms[:7].tolist()
            )
            plan_args['transformation'] = util.pose_from_matrix(state.transformation)
            plan_args['N'] = self.calc_n(state.transformation[0, -1],
                                         state.transformation[1, -1])

            # get primitive plan
            nominal_plan = self.get_nominal_plan(plan_args)

        subplan_tip_poses = nominal_plan[0]['palm_poses_world']

        # setup motion planning request with cartesian waypoints
        tip_right, tip_left = [], []

        # create an approach waypoint near the object
        pre_pose_right_init = util.unit_pose()
        pre_pose_left_init = util.unit_pose()

        pre_pose_right_init.pose.position.y += 0.05
        pre_pose_left_init.pose.position.y += 0.05

        pre_pose_right = util.transform_pose(pre_pose_right_init,
                                             subplan_tip_poses[0][1])
        pre_pose_left = util.transform_pose(pre_pose_left_init,
                                            subplan_tip_poses[0][0])
        tip_right.append(pre_pose_right.pose)
        tip_left.append(pre_pose_left.pose)

        # create all other cartesian waypoints
        for i in range(len(subplan_tip_poses)):
            tip_right.append(subplan_tip_poses[i][1].pose)
            tip_left.append(subplan_tip_poses[i][0].pose)

        if start_joints is None:
            r_start = self.start_joints_r
            l_start = self.start_joint_l
        else:
            r_start = start_joints['right']
            l_start = start_joints['left']

        # l_start = self.robot.get_jpos(arm='left')

        # plan cartesian path
        valid = False
        try:
            # self.robot.mp_right.plan_waypoints(
            #     tip_right,
            #     force_start=l_start+r_start,
            #     avoid_collisions=False
            # )
            self.mp_func(
                tip_right,
                tip_left,
                force_start=l_start+r_start
            )
            valid = True
        except ValueError:
            pass

        return valid

    def mp_func(self, tip_right, tip_left, force_start):
        self.robot.mp_right.plan_waypoints(
            tip_right,
            force_start=force_start,
            avoid_collisions=self.avoid_collisions
        )


class PullLeftSkill(PullRightSkill):
    def __init__(self, sampler, robot, get_plan_func, ignore_mp=False, avoid_collisions=True):
        super(PullLeftSkill, self).__init__(sampler, robot, get_plan_func, ignore_mp, avoid_collisions)

    def sample(self, state, *args, **kwargs):
        final_trans = False
        if 'final_trans' in kwargs.keys():
            final_trans = kwargs['final_trans']
        if final_trans:
            final_trans_to_go = state.transformation_to_go
        else:
            final_trans_to_go = None

        pcd_pts = copy.deepcopy(state.pointcloud)
        pcd_pts[:, 1] = -pcd_pts[:, 1]
        pcd_pts_full = None
        if state.pointcloud_full is not None:
            pcd_pts_full = copy.deepcopy(state.pointcloud_full)
            pcd_pts_full[:, 1] = -pcd_pts_full[:, 1]

        prediction = self.sampler.sample(
            pcd_pts,
            state_full=pcd_pts_full,
            final_trans_to_go=final_trans_to_go)

        if final_trans:
            # on last step we have to trust that this is correct
            new_transformation = prediction['transformation']
        else:
            # if in the middle, then flip based on the right pull transform
            new_transformation = copy.deepcopy(prediction['transformation'])
            new_transformation[0, 1] *= -1
            new_transformation[1, 0] *= -1
            new_transformation[1, -1] *= -1
        new_palms = util.pose_stamped2np(util.flip_palm_pulling(util.list2pose_stamped(prediction['palms'][:7])))
        new_palms[1] *= -1

        new_state = PointCloudNode()
        new_state.init_state(state, new_transformation)
        # new_state.init_palms(new_palms)
        new_state.init_palms(prediction['palms'],
                             correction=True,
                             prev_pointcloud=state.pointcloud_full,
                             dual=False)          
        return new_state

    def mp_func(self, tip_right, tip_left, force_start):
        self.robot.mp_left.plan_waypoints(
            tip_left,
            force_start=force_start,
            avoid_collisions=self.avoid_collisions
        )