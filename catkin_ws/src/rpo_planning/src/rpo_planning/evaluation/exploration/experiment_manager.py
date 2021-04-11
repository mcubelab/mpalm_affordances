import os
import os.path as osp
import time
import numpy as np
import threading
from multiprocessing import Process, Pipe
import pickle
import copy
import pybullet as p

from rpo_planning.utils import common as util


class SimpleRPOEvalManager(object):
    def __init__(self, data_dir, exp_name, cfg):
        self.monitored_object_id = None
        self.cfg = cfg

        self.data_dir = data_dir
        self.exp_name = exp_name

        self.global_data = []
        self.object_data = None
        self.global_trials = 0

    def set_object_id(self, obj_id, obj_fname):
        self.monitored_object_id = obj_id
        self.object_data = {}
        self.object_data['obj_name'] = obj_fname
        self.object_data['trials'] = 0
        self.object_data['mp_success'] = 0
        self.object_data['mp_attempts'] = []
        self.object_data['planning_time'] = 0
        self.object_data['skeleton'] = None
        self.object_data['planning_failure'] = []
        self.object_data['flags'] = None

    def get_object_data(self):
        data_copy = copy.deepcopy(self.object_data)
        return data_copy

    def set_mp_success(self, mp_success, attempts):
        self.object_data['mp_success'] += mp_success
        self.object_data['mp_attempts'].append(attempts)
        self.global_trials += 1
    
    def set_surface_contact_data(self, surf_contact_data_dict):
        self.object_data['surface_contact_data'] = surf_contact_data_dict



class RPOEvalManager(object):
    def __init__(self, robot, pb_client, data_dir, exp_name, cfg):
        self.pos_thresh = 0.005
        self.ori_thresh = 0.01
        self.pull_n_thresh = 1.0
        self.grasp_n_thresh = 1.0
        self.noise_variance = 0.0
        self.palm_corrections = False

        self.monitored_object_id = None

        self.robot = robot
        self.pb_client = pb_client
        self.cfg = cfg

        self.trial_start = False
        self.trial_end = False
        self.trial_is_running = False
        self.trial_timeout = 60.0
        self.on_table_pos_thresh = 0.8
        self.data_dir = data_dir
        self.exp_name = exp_name

        self.global_data = []
        self.object_data = None

    def set_object_id(self, obj_id, obj_fname):
        self.monitored_object_id = obj_id
        self.object_data = {}
        self.object_data['obj_name'] = obj_fname
        self.object_data['execution_finished'] = False
        self.object_data['grasp_duration'] = []
        self.object_data['trial_duration'] = []
        self.object_data['grasp_success'] = []
        self.object_data['final_pos_error'] = []
        self.object_data['final_ori_error'] = []
        self.object_data['pos_success'] = 0
        self.object_data['ori_success'] = 0
        self.object_data['face_success'] = 0
        self.object_data['lost_object'] = []
        self.object_data['trials'] = 0
        self.object_data['mp_success'] = 0
        self.object_data['mp_attempts'] = []
        self.object_data['planning_time'] = 0
        self.object_data['skeleton'] = None
        self.object_data['execute_success'] = 0
        self.object_data['predictions'] = []
        self.object_data['camera_inds'] = []
        self.object_data['camera_noise'] = []
        self.object_data['planning_failure'] = []
        self.object_data['flags'] = None

    def get_object_data(self):
        data_copy = copy.deepcopy(self.object_data)
        filtered_pos_err, filtered_ori_err = self.filter_pose_error_contact_success()
        data_copy['final_pos_error_filtered'] = filtered_pos_err
        data_copy['final_ori_error_filtered'] = filtered_ori_err
        return data_copy

    def get_global_data(self):
        return self.global_data

    def record_object_data(self):
        if self.object_data is not None:
            self.global_data.append(self.object_data)

    def check_pose_success(self, pos_err, ori_err):
        return pos_err < self.pos_thresh, ori_err < self.ori_thresh

    def set_mp_success(self, mp_success, attempts):
        self.object_data['mp_success'] += mp_success
        self.object_data['mp_attempts'].append(attempts)

    def set_planning_failure(self, failure_data):
        self.object_data['planning_failure'].append(failure_data)

    def set_execute_success(self, exec_success):
        self.object_data['execute_success'] = exec_success

    @staticmethod
    def compute_pose_error(trial_data):
        real_final_pose = trial_data['final_pose']
        desired_final_pose = trial_data['goal_pose']

        pos_err, ori_err = util.pose_difference_np(real_final_pose,
                                                   desired_final_pose)
        return pos_err, ori_err

    def filter_pose_error_contact_success(self):
        contact_success_np = np.asarray(self.object_data['grasp_success'])
        contact_success_inds = np.where(contact_success_np)[0]

        if len(self.object_data['final_pos_error']) > 0:
            filtered_pos_err = np.asarray(self.object_data['final_pos_error'])[contact_success_inds].tolist()
            filtered_ori_err = np.asarray(self.object_data['final_ori_error'])[contact_success_inds].tolist()
        else:
            filtered_pos_err, filtered_ori_err = [], []
        return filtered_pos_err, filtered_ori_err

    def end_trial(self, trial_data, grasp_success):
        self.object_data['trials'] += 1
        # check if correct face ended in contact with the table
        if trial_data is not None:
            # check how far final pose was from desired pose
            pos_err, ori_err = self.compute_pose_error(trial_data)
            pos_success, ori_success = self.check_pose_success(pos_err,
                                                               ori_err)

            # write and save data
            self.object_data['grasp_success'].append(grasp_success)

            self.object_data['final_pos_error'].append(pos_err)
            self.object_data['final_ori_error'].append(ori_err)
            self.object_data['pos_success'] += pos_success
            self.object_data['ori_success'] += ori_success

            if 'planning_time' in trial_data.keys():
                self.object_data['planning_time'] = trial_data['planning_time']
            if 'predictions' in trial_data.keys():
                self.object_data['predictions'].append(trial_data['predictions'])
            if 'camera_inds' in trial_data.keys():
                self.object_data['camera_inds'].append(trial_data['camera_inds'])
            if 'camera_noise' in trial_data.keys():
                self.object_data['camera_noise'].append(trial_data['camera_noise'])
            if 'planning_failure' in trial_data.keys():
                self.object_data['planning_failure'].append(trial_data['planning_failure'])
            if 'flags' in trial_data.keys():
                self.object_data['flags'] = trial_data['flags']
