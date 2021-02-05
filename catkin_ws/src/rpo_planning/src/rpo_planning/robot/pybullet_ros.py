import os
import time
import argparse
import numpy as np
import copy
import threading
import trimesh

import rospy
from scipy.interpolate import UnivariateSpline
from pykdl_utils.kdl_kinematics import KDLKinematics
from urdf_parser_py.urdf import URDF
from trac_ik_python import trac_ik
from geometry_msgs.msg import PoseStamped, Pose
import tf.transformations as transformations
import moveit_commander

from planning import pushing_planning, grasp_planning
from planning import levering_planning, pulling_planning
from helper import util, collisions
from motion_planning.group_planner import GroupPlanner

import pybullet as p
from airobot import Robot
from airobot.utils import pb_util, common

from rpo_planning.robot.yumi_ar_ros import YumiAIRobotROS
from example_config_cfg import get_cfg_defaults


class YumiPybullet(object):
    """
    Class for interfacing with Yumi in PyBullet
    with external motion planning, inverse kinematics,
    and forward kinematics, along with other helpers
    """
    def __init__(self, yumi_ar, cfg, exec_thread=True, sim_step_repeat=10):
        """
        Class constructor. Sets up internal motion planning interface
        for each arm, forward and inverse kinematics solvers, and background
        threads for updating the position of the robot.

        Args:
            yumi_pb (airobot Robot): Instance of PyBullet simulated robot, from
                airobot library
            cfg (YACS CfgNode): Configuration parameters
            exec_thread (bool, optional): Whether or not to start the
                background joint position control thread. Defaults to True.
            sim_step_repeat (int, optional): Number of simulation steps
                to take each time the desired joint position value is
                updated. Defaults to 10
        """
        super(YumiPybullet, self).__init__(yumi_ar, cfg)
        self.sim_step_repeat = sim_step_repeat

        self.joint_lock = threading.RLock()
        self._both_pos = self.yumi_pb.arm.get_jpos()
        self._single_pos = {}
        self._single_pos['right'] = \
            self.yumi_pb.arm.arms['right'].get_jpos()
        self._single_pos['left'] = \
            self.yumi_pb.arm.arms['left'].get_jpos()

        self.execute_thread = threading.Thread(target=self._execute_both)
        self.execute_thread.daemon = True
        if exec_thread:
            self.execute_thread.start()
            self.step_sim_mode = False
        else:
            self.step_sim_mode = True

    def _execute_single(self):
        """
        Background thread for controlling a single arm
        """
        while True:
            self.joint_lock.acquire()
            self.yumi_pb.arm.set_jpos(self._both_pos, wait=True)
            self.joint_lock.release()
            time.sleep(0.01)

    def _execute_both(self):
        """
        Background thread for controlling both arms
        """
        while True:
            self.joint_lock.acquire()
            self.yumi_pb.arm.set_jpos(self._both_pos, wait=True)
            self.joint_lock.release()
            time.sleep(0.01)

    def update_joints(self, pos, arm=None):
        """
        Setter function for external user to update the target
        joint values for the arms. If manual step mode is on,
        this function also takes simulation steps.

        Args:
            pos (list): Desired joint angles, either for both arms or
                a single arm
            arm (str, optional): Which arm to update the joint values for
                either 'right', or 'left'. If none, assumed updating for
                both. Defaults to None.

        Raises:
            ValueError: Bad arm name
        """
        if arm is None:
            self.joint_lock.acquire()
            self._both_pos = pos
            self.joint_lock.release()
        elif arm == 'right':
            both_pos = list(pos) + self.yumi_pb.arm.arms['left'].get_jpos()
            self.joint_lock.acquire()
            self._both_pos = both_pos
            self.joint_lock.release()
        elif arm == 'left':
            both_pos = self.yumi_pb.arm.arms['right'].get_jpos() + list(pos)
            self.joint_lock.acquire()
            self._both_pos = both_pos
            self.joint_lock.release()
        else:
            raise ValueError('Arm not recognized')
        self.yumi_pb.arm.set_jpos(self._both_pos, wait=False)
        if self.step_sim_mode:
            for _ in range(self.sim_step_repeat):
                # step_simulation()
                self.yumi_pb.pb_client.stepSimulation()