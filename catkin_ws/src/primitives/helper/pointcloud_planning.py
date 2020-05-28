import os, sys
import os.path as osp
import pickle
import numpy as np

import trimesh
import open3d

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
from eval_utils.visualization_tools import correct_grasp_pos
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

# sys.path.append('/root/training/gat/')
# from models_vae import GoalVAE, GeomVAE, JointVAE

# sys.path.append('/root/training/gat/')
# from models_vae import JointVAE


class PointCloudNode(object):
    def __init__(self):
        self.parent = None
        self.pointcloud = None
        self.pointcloud_full = None
        self.pointcloud_mask = None
        self.transformation = None
        self.transformation_to_go = np.eye(4)
        self.transformation_so_far = None
        self.palms = None
        self.palms_corrected = None
        self.palms_raw = None

    def set_pointcloud(self, pcd, pcd_full=None, pcd_mask=None):
        self.pointcloud = pcd
        self.pointcloud_full = pcd_full
        self.pointcloud_mask = pcd_mask

    def set_trans_to_go(self, trans):
        self.transformation_to_go = trans

    def init_state(self, state, transformation):
        # compute the pointcloud based on the previous pointcloud and specified trans
        pcd_homog = np.ones((state.pointcloud.shape[0], 4))
        pcd_homog[:, :-1] = state.pointcloud
        self.pointcloud = np.matmul(transformation, pcd_homog.T).T[:, :-1]

        # do the same for the full pointcloud, if it's there
        if state.pointcloud_full is not None:
            pcd_full_homog = np.ones((state.pointcloud_full.shape[0], 4))
            pcd_full_homog[:, :-1] = state.pointcloud_full
            self.pointcloud_full = np.matmul(transformation, pcd_full_homog.T).T[:, :-1]

        # node's one step transformation
        self.transformation = transformation

        # transformation to go based on previous transformation to go
        self.transformation_to_go = np.matmul(state.transformation_to_go, np.linalg.inv(transformation))

        # transformation so far, accounting for if this is the first step,
        # and parent node has no transformation so far
        if state.transformation is not None:
            self.transformation_so_far = np.matmul(transformation, state.transformation_so_far)
        else:
            self.transformation_so_far = transformation

    def init_palms(self, palms, correction=False, prev_pointcloud=None):
        if correction and prev_pointcloud is not None:
            palms_raw = palms
            palms_positions = {}
            palms_positions['right'] = palms_raw[:3]
            palms_positions['left'] = palms_raw[7:7+3]
            pcd_pts = prev_pointcloud
            palms_positions_corr = correct_grasp_pos(palms_positions,
                                                     pcd_pts)
            palm_right_corr = np.hstack([
                palms_positions_corr['right'],
                palms_raw[3:7]])
            palm_left_corr = np.hstack([
                palms_positions_corr['left'],
                palms_raw[7+3:]
            ])
            self.palms_corrected = np.hstack([palm_right_corr, palm_left_corr])
            self.palms_raw = palms_raw
            self.palms = self.palms_corrected
        else:
            self.palms = palms


class PointCloudTree(object):
    def __init__(self, start_pcd, trans_des, skeleton, skills,
                 start_pcd_full=None, target=None, motion_planning=True,
                 only_rot=True):
        self.skeleton = skeleton
        self.skills = skills
        self.goal_threshold = None
        self.motion_planning = motion_planning
        self.timeout = 300
        self.only_rot = only_rot

        self.pos_thresh = 0.005
        self.ori_thresh = 0.01
        self.eye_thresh = 1.0

        self.buffers = {}
        for i in range(len(skeleton)):
            self.buffers[i+1] = []

        self.start_node = PointCloudNode()
        self.start_node.set_pointcloud(start_pcd, start_pcd_full)
        self.start_node.set_trans_to_go(trans_des)
        self.transformation = np.eye(4)

        self.buffers[0] = [self.start_node]

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
                        valid = self.skills[self.skeleton[i+1]].satisfies_preconditions(sample)

                        # check if this is a valid transition (motion planning)
                        if valid:
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

                    if not self.skills[skill].valid_transformation(sample):
                        # pop sample that this came from
                        self.buffers[i].pop(index)
                        continue

                    # still check motion planning for final step
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
                final_trans=last_step)
            index = 0
        else:
            # sample from the buffers we have
            if len(self.buffers[i]) > 0:
                index = np.random.randint(len(self.buffers[i]))
                state = self.buffers[i][index]
                sample = self.skills[skill].sample(
                    state,
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


class GraspSamplerVAE(object):
    def __init__(self, model, default_target, latent_dim=256):
        self.dev = torch.device('cuda')
        self.model = model.eval().to(self.dev)
        self.latent_dim = latent_dim
        self.kd_idx = torch.from_numpy(np.arange(100))
        self.default_target = default_target

    def get_transformation(self, state, mask, target=None, final_trans_to_go=None):
        source = state[np.where(mask)[0], :]
        source_obj = state
        if target is None:
            target = self.default_target

        init_trans_fwd = reg.init_grasp_trans(source, fwd=True)
        init_trans_bwd = reg.init_grasp_trans(source, fwd=False)

        if final_trans_to_go is None:
            init_trans = init_trans_fwd
        else:
            init_trans = final_trans_to_go
        transform = copy.deepcopy(reg.full_registration_np(source, target, init_trans))
        source_obj_trans = reg.apply_transformation_np(source_obj, transform)

#         if np.where(source_obj_trans[:, 2] < 0.005)[0].shape[0] > 100:
        if np.mean(source_obj_trans, axis=0)[2] < 0.005:
            init_trans = init_trans_bwd
            transform = copy.deepcopy(reg.full_registration_np(source, target, init_trans))
            source_obj_trans = reg.apply_transformation_np(source_obj, transform)

        return transform

    def sample(self, state, target=None, final_trans_to_go=None):
        state = state[:100]
        state_np = state
        state_mean = np.mean(state, axis=0)
        state_normalized = (state - state_mean)
        state_mean = np.tile(state_mean, (state.shape[0], 1))

        state_full = np.concatenate([state_normalized, state_mean], axis=1)

        mask_predictions = []
        palm_predictions = []
        joint_keypoint = torch.from_numpy(state_full)[None, :, :].float().to(self.dev)
        for repeat in range(10):
            palm_repeat = []
            z = torch.randn(1, self.latent_dim).to(self.dev)
            recon_mu, ex_wt = self.model.decode(z, joint_keypoint)
            output_r, output_l, pred_mask, pred_trans = recon_mu
            mask_predictions.append(pred_mask.detach().cpu().numpy())

            output_r, output_l = output_r.detach().cpu().numpy(), output_l.detach().cpu().numpy()
            output_joint = np.concatenate([output_r, output_l], axis=2)
            ex_wt = ex_wt.detach().cpu().numpy().squeeze()
            # sort_idx = np.argsort(ex_wt, axis=1)[:, ::-1]
            sort_idx = np.argsort(ex_wt)[None, :]

            for i in range(output_joint.shape[0]):
                for j in range(output_joint.shape[1]):
                    j = sort_idx[i, j]
                    pred_info = output_joint[i, j]
            #         pred_info = obj_frame[i].cpu().numpy()
                    palm_repeat.append(pred_info.tolist())
            palm_predictions.append(palm_repeat)
        palm_predictions = np.asarray(palm_predictions).squeeze()
        mask_predictions = np.asarray(mask_predictions).squeeze()

        mask_ind = np.random.randint(10)
        palm_ind = np.random.randint(5)
        pred_mask = mask_predictions[mask_ind]
        pred_palm = palm_predictions[mask_ind, palm_ind, :]

        top_inds = np.argsort(pred_mask)[::-1]
        pred_mask = np.zeros((pred_mask.shape[0]), dtype=bool)
        pred_mask[top_inds[:15]] = True

        prediction = {}
        prediction['palms'] = pred_palm
        prediction['mask'] = pred_mask
        prediction['transformation'] = self.get_transformation(state_np, pred_mask, target)
        return prediction


class GraspSamplerVAEPubSub(object):
    def __init__(self, default_target, obs_dir, pred_dir, pointnet=False):
        self.default_target = default_target
        self.obs_dir = obs_dir
        self.pred_dir = pred_dir
        self.samples_count = 0
        self.pointnet = pointnet

    def update_default_target(self, target):
        self.default_target = target

    def get_transformation(self, state, mask,
                           target=None, final_trans_to_go=None):
        source = state[np.where(mask)[0], :]
        source_obj = state
        if target is None:
            target = self.default_target

        init_trans_fwd = reg.init_grasp_trans(source, fwd=True)
        init_trans_bwd = reg.init_grasp_trans(source, fwd=False)

        if final_trans_to_go is None:
            init_trans = init_trans_fwd
        else:
            init_trans = final_trans_to_go
        transform = copy.deepcopy(reg.full_registration_np(source, target, init_trans))
        source_obj_trans = reg.apply_transformation_np(source_obj, transform)

        if np.mean(source_obj_trans, axis=0)[2] < 0.005:
            init_trans = init_trans_bwd
            transform = copy.deepcopy(reg.full_registration_np(source, target, init_trans))
            source_obj_trans = reg.apply_transformation_np(source_obj, transform)

        return transform

    def sample(self, state, target=None, final_trans_to_go=None):
        self.samples_count += 1
        # put inputs inside numpy file
        T_mat = np.eye(4)
        transformation = np.asarray(util.pose_stamped2list(util.pose_from_matrix(T_mat)), dtype=np.float32)
        pointcloud_pts = state[:100]

        # write to known location
        obs_fname = os.path.join(self.obs_dir, str(self.samples_count) + '.npz')
        np.savez(
            obs_fname,
            pointcloud_pts=pointcloud_pts,
            transformation=transformation
        )

        # wait for return
        got_file = False
        pred_fname = os.path.join(self.pred_dir, str(self.samples_count) + '.npz')
        start = time.time()
        while True:
            try:
                prediction = np.load(pred_fname)
                got_file = True
            except:
                pass
            if got_file or (time.time() - start > 300):
                break
            time.sleep(0.01)
        # if not got_file:
        #     wait = raw_input('waiting for predictions to come back online')
        #     return
        os.remove(pred_fname)

        # unpack from returned file
        ind = np.random.randint(prediction['mask_predictions'].shape[0])
        ind_contact = np.random.randint(5)

        mask = prediction['mask_predictions'][ind, :]
        top_inds = np.argsort(mask)[::-1]
        pred_mask = np.zeros((mask.shape[0]), dtype=bool)
        pred_mask[top_inds[:15]] = True

        # embed()

        # fix palm predictions (via CoM)
        if self.pointnet:
            contact_r = prediction['palm_predictions'][ind, :7]
            contact_l = prediction['palm_predictions'][ind, 7:]
        else:
            contact_r = prediction['palm_predictions'][ind, ind_contact, :7]
            contact_l = prediction['palm_predictions'][ind, ind_contact, 7:]

        contact_r[:3] += np.mean(pointcloud_pts, axis=0)
        contact_l[:3] += np.mean(pointcloud_pts, axis=0)

        # put into local prediction
        prediction_dict = {}
        prediction_dict['palms'] = np.hstack([contact_r, contact_l])
        prediction_dict['mask'] = pred_mask
        prediction_dict['transformation'] = self.get_transformation(
            pointcloud_pts,
            pred_mask,
            target,
            final_trans_to_go)
        return prediction_dict


class PullSamplerBasic(object):
    def __init__(self):
        self.x_bounds = [0.1, 0.5]
        self.y_bounds = [-0.4, 0.4]
        self.theta_bounds = [-np.pi, np.pi]

        self.rand_pull_yaw = lambda: (np.pi/2)*np.random.random_sample() + np.pi/2

        self.sample_timeout = 5.0
        self.sample_limit = 100

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
#         print(np.rad2deg(theta))
        rot = common.euler2rot([0, 0, theta])
        transform = np.eye(4)
        transform[:-1, :-1] = rot
        transform[:2, -1] = np.array([x-x_pos, y-y_pos])
        return transform

    def get_palms(self, state):
        # check if full pointcloud available, if not use sparse pointcloud
        pcd_pts = state

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

    def sample(self, state=None, final_trans_to_go=None):
        prediction = {}
        prediction['transformation'] = self.get_transformation(state, final_trans_to_go)
        prediction['palms'] = self.get_palms(state)
        return prediction


class PrimitiveSkill(object):
    def __init__(self, sampler, robot):
        """Base class for primitive skill

        Args:
            robot (TODO): Interface to the robot (PyBullet, MoveIt, ROS)
            sampler (function): sampling function that generates new
                potential state to add to the plan
        """
        self.robot = robot
        self.sampler = sampler
        self.table_x_min, self.table_x_max = 0.1, 0.5
        self.table_y_min, self.table_y_max = -0.3, 0.3

    def valid_transformation(self, state):
        raise NotImplementedError

    def satisfies_preconditions(self, state):
        raise NotImplementedError

    def sample(self, state, target_surface=None, final_trans=False):
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
    def __init__(self, sampler, robot, get_plan_func, ignore_mp=False):
        super(GraspSkill, self).__init__(sampler, robot)
        self.x_min, self.x_max = 0.35, 0.45
        self.y_min, self.y_max = -0.1, 0.1
        self.start_joints = [0.9936, -2.1848, -0.9915, 0.8458, 3.7618,  1.5486,  0.1127,
                            -1.0777, -2.1187, 0.995, 1.002 ,  -3.6834,  1.8132,  2.6405]
        self.get_plan_func = get_plan_func
        self.ignore_mp = ignore_mp

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
                state.pointcloud,
                target_surface,
                state.transformation_to_go)
        else:
            prediction = self.sampler.sample(
                state.pointcloud,
                target_surface)
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
                    avoid_collisions=True
                )
                right_valid.append(1)
            except ValueError as e:
                break
            try:
                self.robot.mp_left.plan_waypoints(
                    tip_left,
                    force_start=l_start+r_start,
                    avoid_collisions=True
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
    def __init__(self, sampler, robot, get_plan_func, ignore_mp=False):
        super(PullRightSkill, self).__init__(sampler, robot)
        self.get_plan_func = get_plan_func
        self.start_joints_r = [0.417, -1.038, -1.45, 0.26, 0.424, 1.586, 2.032]
        self.unit_n = 100
        self.ignore_mp = ignore_mp

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

        if state.pointcloud_full is None:
            pcd_pts = state.pointcloud
        else:
            pcd_pts = state.pointcloud_full

        prediction = self.sampler.sample(
            pcd_pts,
            final_trans_to_go=final_trans_to_go)
        new_state = PointCloudNode()
        new_state.init_state(state, prediction['transformation'])
        new_state.init_palms(prediction['palms'])
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
        else:
            r_start = start_joints['right']

        l_start = self.robot.get_jpos(arm='left')

        # plan cartesian path
        valid = False
        try:
            self.robot.mp_right.plan_waypoints(
                tip_right,
                force_start=l_start+r_start,
                avoid_collisions=True
            )
            valid = True
        except ValueError:
            pass

        return valid