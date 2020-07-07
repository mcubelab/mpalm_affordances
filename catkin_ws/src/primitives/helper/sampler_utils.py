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


class PubSubSamplerBase(object):
    def __init__(self, obs_dir, pred_dir, sampler_prefix):
        self.obs_dir = obs_dir
        self.pred_dir = pred_dir
        self.sampler_prefix = sampler_prefix
        self.samples_count = 0

    def filesystem_pub_sub(self, state):
        self.samples_count += 1
        T_mat = np.eye(4)
        transformation = util.pose_stamped2np(util.pose_from_matrix(T_mat))
        pointcloud_pts = state[:100]

        obs_fname = osp.join(
            self.obs_dir,
            self.sampler_prefix + str(self.samples_count) + '.npz')
        np.savez(
            obs_fname,
            pointcloud_pts=pointcloud_pts,
            transformation=transformation
        )

        # wait for return
        got_file = False
        pred_fname = osp.join(
            self.pred_dir,
            self.sampler_prefix + str(self.samples_count) + '.npz')
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
        os.remove(pred_fname)     
        return prediction   