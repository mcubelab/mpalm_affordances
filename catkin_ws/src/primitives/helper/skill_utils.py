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
        # return x_valid and y_valid
        return True