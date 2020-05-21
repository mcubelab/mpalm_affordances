import os
import os.path as osp
import time
import numpy as np
import threading
import pickle
import copy

import pybullet as p
from helper import util2 as util


class GraspEvalManager(object):
    def __init__(self, robot, pb_client, data_dir, exp_name, cfg):
        self.overall_trials = 0
        self.stable_grasp_success = 0
        self.correct_goal_face = 0
        self.within_trans_des_err = 0
        self.lost_object = 0
        self.mp_success = 0

        self.pos_thresh = 0.005
        self.ori_thresh = 0.01
        self.noise_variance = 0.0
        self.palm_corrections = False

        self.monitored_object_id = None

        self.robot = robot
        self.pb_client = pb_client
        self.cfg = cfg

        self.trial_start = False
        self.trial_end = False
        self.trial_is_running = False
        self.data_dir = data_dir
        self.exp_name = exp_name

        self.global_data = []
        self.object_data = None

    def set_object_id(self, obj_id, obj_fname):
        self.monitored_object_id = obj_id
        self.object_data = {}
        self.object_data['obj_name'] = obj_fname

    def record_object_data(self):
        if self.object_data is not None:
            self.global_data.append(self.object_data)

    def start_trial(self):
        self.trial_start = True

    def end_trial(self, trial_data):
        self.trial_end = True
        self.trial_data = trial_data

    def get_trial_data(self):
        return copy.deepcopy(self.trial_data)

    def object_table_contact(self):
        table_contacts = p.getContactPoints(
            self.robot.yumi_pb.arm.robot_id,
            self.monitored_object_id,
            self.cfg.TABLE_ID,
            -1,
            self.pb_client)
        return len(table_contacts) > 0

    def object_palm_contact(self, arm='right'):
        palm_id = self.cfg.RIGHT_GEL_ID if arm == 'right' \
            else self.cfg.LEFT_GEL_ID
        palm_contacts = p.getContactPoints(
            self.robot.yumi_pb.arm.robot_id,
            self.monitored_object_id,
            palm_id,
            -1,
            self.pb_client)
        return len(palm_contacts) > 0

    def object_fly_away(self):
        obj_pos = p.getBasePositionAndOrientation(self.monitored_object_id)[0]
        return obj_pos[2] < -0.1

    def check_pose_success(self, pos_err, ori_err):
        return pos_err < self.pos_thresh, ori_err < self.ori_thresh

    def set_mp_success(self, mp_success, attempts):
        self.object_data['mp_success'] += mp_success
        self.object_data['mp_attempts'].append(attempts)

    @staticmethod
    def check_face_success(trial_data):
        pcd_pts = trial_data['start_pcd_down']
        mask = trial_data['start_pcd_mask']
        transformation = trial_data['trans_executed']

        mask_pts = pcd_pts[np.where(mask)[0], :]

        mask_pts_h = np.ones((mask_pts.shape[0], 4))
        mask_pts_h[:, :-1] = mask_pts
        mask_pts_final = np.matmul(transformation, mask_pts_h.T).T[:, :-1]

        return np.mean(mask_pts_final[:, 2]) < 0.01

    @staticmethod
    def compute_pose_error(trial_data):
        real_final_pose = trial_data['final_pose']
        desired_final_pose = trial_data['goal_pose']

        pos_err, ori_err = util.pose_difference_np(real_final_pose,
                                                   desired_final_pose)
        return pos_err, ori_err

    def monitoring_thread(self):
        while True:
            # know when a trial starts
            if self.trial_start and not self.trial_is_running:
                # setup information for trial
                self.trial_is_running = True
                self.table_contact = True
                self.grasp_started = False

                start_trial_time = time.time()
                done = False
                lost_object = False
                self.overall_trials += 1
                while not done:
                    # check if object is in contact with the table
                    # keep track of how long, to increment stable grasp
                    self.table_contact = self.object_table_contact()
                    self.r_contact = self.object_palm_contact(arm='right')
                    self.l_contact = self.object_palm_contact(arm='left')

                    both_contact = self.r_contact and self.l_contact
                    grasping = both_contact and not self.table_contact

                    if grasping and not self.grasp_started:
                        start_grasp_time = time.time()
                        self.grasp_started = True
                        grasp_success = True

                    if self.grasp_started and not grasping:
                        end_grasp_time = time.time()
                        grasp_success = False

                    if self.object_fly_away():
                        lost_object = True
                        self.lost_object += 1
                        done = True
                        break

                    if self.trial_end:
                        done = True
                        break

                    time.sleep(0.1)
                self.trial_end = False
                self.trial_start = False
                trial_data = self.get_trial_data()

                # grasp success based on grasp end time and trial end time
                end_trial_time = time.time()
                grasp_duration = end_grasp_time - start_grasp_time
                trial_duration = end_trial_time - start_trial_time
                grasp_success = end_trial_time - end_grasp_time < 3.0

                # check if correct face ended in contact with the table
                face_success = self.check_face_success(trial_data)

                # check how far final pose was from desired pose
                pos_err, ori_err = self.compute_pose_error(trial_data)
                pos_success, ori_success = self.check_pose_success(pos_err,
                                                                   ori_err)

                # check how far final pose was from global desired pose
                # global_pose_err = self.compute_global_error(trial_data)
                # global_des_success = global_pose_err < self.trans_des_thresh

                # write and save data
                self.object_data['execution_finished'] = True
                self.object_data['grasp_duration'].append(grasp_duration)
                self.object_data['trial_duration'].append(trial_duration)
                self.object_data['grasp_success'] += grasp_success
                self.object_data['final_pos_error'].append(pos_err)
                self.object_data['final_ori_error'].append(ori_err)
                self.object_data['pos_success'] += pos_success
                self.object_data['ori_success'] += ori_success
                self.object_data['face_success'] += face_success
                self.object_data['lost_object'] += lost_object
                self.object_data['trials'] += 1
        time.sleep(0.1)
