import os
import os.path as osp
import time
import numpy as np
import threading
from multiprocessing import Process, Pipe
import pickle
import copy
import pybullet as p

import util2 as util


class GuardedMover(object):
    def __init__(self, robot, pb_client, cfg, verbose=False):
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

        self.verbose = verbose

    def set_object_id(self, obj_id):
        self.monitored_object_id = obj_id

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
            if self.verbose:
                print('r: ' + str(max(r_n_list)))
            r_contact = r_contact and (max(r_n_list) > self.grasp_n_thresh)
        if l_contact and n:
            if self.verbose:
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
                if self.verbose:
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
                if self.verbose:
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
