import os, os.path as osp
import sys
import time
import argparse
import numpy as np
from multiprocessing import Process, Pipe, Queue
import pickle
import rospy
import copy
import signal
import open3d
import threading
import cv2
import scipy.misc
from PIL import Image
import matplotlib.pyplot as plt
from IPython import embed

from airobot import Robot
from airobot.sensor.camera.rgbdcam_pybullet import RGBDCameraPybullet
from airobot.sensor.camera.rgbdcam_real import RGBDCameraReal
from airobot.utils.ros_util import read_cam_ext
import pybullet as p

from helper import util
from macro_actions import ClosedLoopMacroActions
from yumi_pybullet_ros import YumiGelslimPybullet
from yumi_real_ros import YumiGelslimReal

from closed_loop_eval import SingleArmPrimitives, DualArmPrimitives

from yacs.config import CfgNode as CN
from closed_loop_experiments_cfg import get_cfg_defaults
from data_tools.proc_gen_cuboids import CuboidSampler



class DataManager(object):
    def __init__(self, save_path):
        self.save_path = save_path
        self.pickle_dir = 'pkl'
        self.img_dir = 'img'
        self.pcd_dir = 'pcd'
        self.depth_scale = 1000.0

    def make_paths(self, raw_fname):

        pcd_fname = os.path.join(
            self.save_path,
            raw_fname,
            self.pcd_dir, raw_fname + '_pcd.pkl')

        depth_fnames = [raw_fname + '_depth_%d.png' % j for j in range(3)]
        rgb_fnames = [raw_fname + '_rgb_%d.png' % j for j in range(3)]
        seg_fnames = [raw_fname + '_seg_%d.pkl' % j for j in range(3)]

        if not os.path.exists(os.path.join(self.save_path, raw_fname, self.pickle_dir)):
            os.makedirs(os.path.join(self.save_path, raw_fname, self.pickle_dir))
        if not os.path.exists(os.path.join(self.save_path, raw_fname, self.img_dir)):
            os.makedirs(os.path.join(self.save_path, raw_fname, self.img_dir))
        if not os.path.exists(os.path.join(self.save_path, raw_fname, self.pcd_dir)):
            os.makedirs(os.path.join(self.save_path, raw_fname, self.pcd_dir))

        return pcd_fname, depth_fnames, rgb_fnames, seg_fnames

    def save_observation(self, data_dict, filename):
        raw_fname = filename

        pcd_fname, depth_fnames, rgb_fnames, seg_fnames = self.make_paths(raw_fname)

        pkl_data = {}
        for key in data_dict.keys():
            if not key == 'obs':
                pkl_data[key] = copy.deepcopy(data_dict[key])

        with open(os.path.join(self.save_path, raw_fname, self.pickle_dir, raw_fname+'.pkl'), 'wb') as pkl_f:
            pickle.dump(pkl_data, pkl_f)

        if 'obs' in data_dict.keys():
            # save depth
            for k, fname in enumerate(rgb_fnames):
                rgb_fname = os.path.join(self.save_path, raw_fname, self.img_dir, rgb_fnames[k])
                depth_fname = os.path.join(self.save_path, raw_fname, self.img_dir, depth_fnames[k])
                seg_fname = os.path.join(self.save_path, raw_fname, self.img_dir, seg_fnames[k])

                # save depth
                sdepth = data_dict['obs']['depth'][k] * self.depth_scale
                cv2.imwrite(depth_fname, sdepth.astype(np.uint16))

                # save rgb
                cv2.imwrite(rgb_fname, data_dict['obs']['rgb'][k])

                # save seg
                with open(seg_fname, 'wb') as f:
                    pickle.dump(data_dict['obs']['seg'][k], f, protocol=pickle.HIGHEST_PROTOCOL)

            # save pointcloud arrays as .pkl with high protocol
            pcd_pts = data_dict['obs']['pcd_pts']
            pcd_colors = data_dict['obs']['pcd_colors']
            down_pcd_pts = data_dict['obs']['down_pcd_pts']
            table_pcd_pts = data_dict['obs']['table_pcd_pts']
            table_pcd_colors = data_dict['obs']['table_pcd_colors']

            pcd = {}
            pcd['pts'] = pcd_pts
            pcd['colors'] = pcd_colors
            pcd['down_pts'] = down_pcd_pts
            pcd['table_pts'] = table_pcd_pts
            pcd['table_colors'] = table_pcd_colors
            with open(pcd_fname, 'wb') as f:
                pickle.dump(pcd, f, protocol=pickle.HIGHEST_PROTOCOL)


class MultiBlockManager(object):
    def __init__(self, cuboid_path, cuboid_sampler,
                 robot_id, table_id, r_gel_id, l_gel_id, fname_prefix='test_cuboid_smaller_'):
        self.sampler = cuboid_sampler
        self.cuboid_path = cuboid_path

        self.r_gel_id = r_gel_id
        self.l_gel_id = l_gel_id
        self.robot_id = robot_id
        self.table_id = table_id

        self.gel_ids = [self.r_gel_id, self.l_gel_id]

        self.cuboid_fname_prefix = fname_prefix
        self.setup_block_set()

    def setup_block_set(self):
        self.cuboid_fnames = []
        for fname in os.listdir(self.cuboid_path):
            if fname.startswith(self.cuboid_fname_prefix):
                self.cuboid_fnames.append(os.path.join(self.cuboid_path,
                                                       fname))
        print('Loaded cuboid files: ')
        # print(self.cuboid_fnames)

    def get_cuboid_fname(self):
        ind = np.random.randint(len(self.cuboid_fnames))
        return self.cuboid_fnames[ind]

    def filter_collisions(self, obj_id, goal_obj_id=None):
        for gel_id in self.gel_ids:
            if goal_obj_id is not None:
                p.setCollisionFilterPair(self.robot_id,
                                         goal_obj_id,
                                         gel_id,
                                         -1,
                                         enableCollision=False)
            p.setCollisionFilterPair(self.robot_id,
                                     obj_id,
                                     gel_id,
                                     -1,
                                     enableCollision=True)
        p.setCollisionFilterPair(self.robot_id,
                                 obj_id,
                                 self.table_id,
                                 -1,
                                 enableCollision=True)
        if goal_obj_id is not None:
            for jnt_id in range(self.table_id):
                p.setCollisionFilterPair(self.robot_id, goal_obj_id, jnt_id, -1, enableCollision=False)
            p.setCollisionFilterPair(self.robot_id,
                                     goal_obj_id,
                                     self.table_id,
                                     -1,
                                     enableCollision=True)

    def robot_collisions_filter(self, obj_id, enable=True):
        for jnt_id in range(self.table_id):
            p.setCollisionFilterPair(self.robot_id, obj_id, jnt_id, -1, enableCollision=enable)
        p.setCollisionFilterPair(self.robot_id,
                                    obj_id,
                                    self.table_id,
                                    -1,
                                    enableCollision=True)

