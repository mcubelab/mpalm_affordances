import os
import os.path as osp
import time
import numpy as np
import threading
from multiprocessing import Process, Pipe
import pickle
import copy
import pybullet as p

from helper import util2 as util


class GraspEvalManager(object):
    def __init__(self, robot, pb_client, data_dir, exp_name,
                 parent, child, work_queue, result_queue, cfg):
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

        self.parent = parent
        self.child = child
        self.work_queue = work_queue
        self.result_queue = result_queue

        # self.monitor_process = Process(
        #     target=self.monitoring_thread,
        #     args=(
        #         self.child,
        #         self.work_queue,
        #         self.result_queue,
        #         self.robot.yumi_pb.arm.robot_id,
        #         self.pb_client,
        #         self.cfg,
        #         self.trial_timeout
        #     )
        # )
        # self.monitor_process.start()

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

    def get_trial_data(self):
        return copy.deepcopy(self.trial_data)

    def check_pose_success(self, pos_err, ori_err):
        return pos_err < self.pos_thresh, ori_err < self.ori_thresh

    def set_mp_success(self, mp_success, attempts):
        self.object_data['mp_success'] += mp_success
        self.object_data['mp_attempts'].append(attempts)

    def set_execute_success(self, exec_success):
        self.object_data['execute_success'] = exec_success

    def still_grasping(self, n=True):
        table_contact, table_n_list = self.object_table_contact(
            self.robot.yumi_pb.arm.robot_id,
            self.monitored_object_id,
            self.cfg.TABLE_ID,
            self.pb_client
        )

        r_contact, r_n_list = self.object_palm_contact(
            self.robot.yumi_pb.arm.robot_id,
            self.monitored_object_id,
            self.cfg.RIGHT_GEL_ID,
            self.pb_client
        )

        l_contact, l_n_list = self.object_palm_contact(
            self.robot.yumi_pb.arm.robot_id,
            self.monitored_object_id,
            self.cfg.LEFT_GEL_ID,
            self.pb_client
        )
        if r_contact and n:
            print('r: ' + str(max(r_n_list)))
            r_contact = r_contact and (max(r_n_list) > self.grasp_n_thresh)
        if l_contact and n:
            print('l: ' + str(max(l_n_list)))
            l_contact = l_contact and (max(l_n_list) > self.grasp_n_thresh)

        return r_contact and l_contact

    def still_pulling(self, n=True, arm='right'):
        if arm == 'right':
            r_contact, r_n_list = self.object_palm_contact(
                self.robot.yumi_pb.arm.robot_id,
                self.monitored_object_id,
                self.cfg.RIGHT_GEL_ID,
                self.pb_client
            )
            if r_contact and n:
                print('r: ' + str(max(r_n_list)))
                r_contact = r_contact and (max(r_n_list) > self.pull_n_thresh)

            return r_contact
        else:
            l_contact, l_n_list = self.object_palm_contact(
                self.robot.yumi_pb.arm.robot_id,
                self.monitored_object_id,
                self.cfg.LEFT_GEL_ID,
                self.pb_client
            )
            if l_contact and n:
                print('r: ' + str(max(l_n_list)))
                l_contact = l_contact and (max(l_n_list) > self.pull_n_thresh)

            return l_contact


    @staticmethod
    def object_fly_away(obj_id):
        obj_pos = p.getBasePositionAndOrientation(obj_id)[0]
        return obj_pos[2] < -0.1

    @staticmethod
    def object_table_contact(robot_id, obj_id, table_id, pb_cl):
        table_contacts = p.getContactPoints(
            robot_id,
            obj_id,
            table_id,
            -1,
            pb_cl)

        n_list = []
        for pt in table_contacts:
            n_list.append(pt[-5])
        return len(table_contacts) > 0, n_list

    @staticmethod
    def object_palm_contact(robot_id, obj_id, palm_id, pb_cl):
        palm_contacts = p.getContactPoints(
            robot_id,
            obj_id,
            palm_id,
            -1,
            pb_cl)

        n_list = []
        for pt in palm_contacts:
            n_list.append(pt[-5])
        return len(palm_contacts) > 0, n_list

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
        # grasp_duration = monitor_result['grasp_duration']
        # trial_duration = monitor_result['trial_duration']
        # grasp_success = monitor_result['grasp_success']
        # lost_object = monitor_result['lost_object']

        self.object_data['trials'] += 1
        # check if correct face ended in contact with the table
        if trial_data is not None:
            # face_success = self.check_face_success(trial_data)

            # check how far final pose was from desired pose
            pos_err, ori_err = self.compute_pose_error(trial_data)
            pos_success, ori_success = self.check_pose_success(pos_err,
                                                               ori_err)

            # check how far final pose was from global desired pose
            # global_pose_err = self.compute_global_error(trial_data)
            # global_des_success = global_pose_err < self.trans_des_thresh

            # write and save data
            # self.object_data['grasp_duration'].append(grasp_duration)
            # self.object_data['trial_duration'].append(trial_duration)
            # self.object_data['grasp_success'] += grasp_success
            self.object_data['grasp_success'].append(grasp_success)
            # self.object_data['face_success'] += face_success

            self.object_data['final_pos_error'].append(pos_err)
            self.object_data['final_ori_error'].append(ori_err)
            self.object_data['pos_success'] += pos_success
            self.object_data['ori_success'] += ori_success

            if pos_err < self.on_table_pos_thresh:
                # self.object_data['final_pos_error'].append(pos_err)
                # self.object_data['final_ori_error'].append(ori_err)
                # self.object_data['pos_success'] += pos_success
                # self.object_data['ori_success'] += ori_success
                self.object_data['lost_object'].append((False, pos_err, ori_err))
            else:
                self.object_data['lost_object'].append((True, pos_err, ori_err))
            # self.object_data['lost_object'] += lost_object

            if 'planning_time' in trial_data.keys():
                self.object_data['planning_time'] = trial_data['planning_time']
            if 'predictions' in trial_data.keys():
                self.object_data['predictions'].append(trial_data['predictions'])
