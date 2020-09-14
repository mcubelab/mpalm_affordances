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
        self.start_joints_r = [0.413, -1.325, -1.040, -0.053, -0.484, 0.841, -1.546]
        self.start_joints_l = [-0.473, -1.450, 1.091, 0.031, 0.513, 0.77, -1.669]        
        self.sv_checker = StateValidity()

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

    def start_goal_validity(self, state):
        """Check if PointCloudNode state is valid at the beginning and end of a skill

        Args:
            state (PointCloudNode): State to check validity for, that includes palm poses and
                subgoal transformation

        Returns:
            2-element tuple containing: 
            - bool: Start state validity (True if valid)
            - bool: Goal state validity (True if valid)
        """
        # get palms and transformation     
        palms_start = state.palms
        transformation = state.transformation
        transformation_pose = util.pose_from_matrix(transformation)

        palm_pose_right_start = util.list2pose_stamped(palms_start[:7])
        palm_pose_right_goal = util.transform_pose(palm_pose_right_start, transformation_pose)

        palm_pose_right_start_np = util.pose_stamped2np(palm_pose_right_start)
        palm_pose_right_goal_np = util.pose_stamped2np(palm_pose_right_goal)        

        # get joint configurations
        joints_right_start = self.robot.compute_ik(
            pos=palm_pose_right_start_np[:3],
            ori=palm_pose_right_start_np[3:],
            seed=self.start_joints_r, 
            arm='right')

        joints_right_goal = self.robot.compute_ik(
            pos=palm_pose_right_goal_np[:3],
            ori=palm_pose_right_goal_np[3:],
            seed=self.start_joints_r, 
            arm='right')

        if joints_right_start is not None:
            r_valid_start = self.robot.mp_right.get_state_validity(joints_right_start)
        else:
            r_valid_start = False         
        if joints_right_goal is not None:
            r_valid_goal = self.robot.mp_right.get_state_validity(joints_right_goal)
        else:
            r_valid_goal = False

        l_valid_start = True
        l_valid_goal = True        
        if palms_start.shape[0] > 7:
            palm_pose_left_start = util.list2pose_stamped(palms_start[7:])
            palm_pose_left_goal = util.transform_pose(palm_pose_left_start, transformation_pose)

            palm_pose_left_start_np = util.pose_stamped2np(palm_pose_left_start)
            palm_pose_left_goal_np = util.pose_stamped2np(palm_pose_left_goal) 

            # get joint configurations
            joints_left_start = self.robot.compute_ik(
                pos=palm_pose_left_start_np[:3],
                ori=palm_pose_left_start_np[3:],
                seed=self.start_joints_l, 
                arm='left')

            joints_left_goal = self.robot.compute_ik(
                pos=palm_pose_left_goal_np[:3],
                ori=palm_pose_left_goal_np[3:],
                seed=self.start_joints_l, 
                arm='left')

            if joints_left_start is not None:
                l_valid_start = self.robot.mp_left.get_state_validity(joints_left_start)
            else:
                l_valid_start = False
            if joints_left_goal is not None:
                l_valid_goal = self.robot.mp_left.get_state_validity(joints_left_goal)
            else:
                l_valid_goal = False
        
        start_valid = r_valid_start and l_valid_start
        goal_valid = r_valid_goal and l_valid_goal
        return start_valid, goal_valid        


import rospy
from moveit_msgs.srv import GetStateValidity, GetStateValidityRequest, GetStateValidityResponse

DEFAULT_SV_SERVICE = "/check_state_validity"

class StateValidity():
    def __init__(self):
        rospy.loginfo("Initializing StateValidity class")
        self.sv_srv = rospy.ServiceProxy("/check_state_validity", GetStateValidity)
        rospy.loginfo("Connecting to State Validity service")
        rospy.wait_for_service("check_state_validity")
        rospy.loginfo("Reached this point")

        if rospy.has_param('/play_motion/approach_planner/planning_groups'):
            list_planning_groups = rospy.get_param('/play_motion/approach_planner/planning_groups')
        else:
            rospy.logwarn("Param '/play_motion/approach_planner/planning_groups' not set. We can't guess controllers")
        rospy.loginfo("Ready for making Validity calls")


    def close_sv(self):
        self.sv_srv.close()


    def get_state_validity(self, robot_state, group_name='both_arms', constraints=None, print_depth=False):
        """Given a RobotState and a group name and an optional Constraints
        return the validity of the State"""
        gsvr = GetStateValidityRequest()
        gsvr.robot_state = robot_state
        gsvr.group_name = group_name
        if constraints != None:
            gsvr.constraints = constraints
        result = self.sv_srv.call(gsvr)

        if (not result.valid):
            contact_depths = []
            for i in range(len(result.contacts)):
                contact_depths.append(result.contacts[i].depth)

            max_depth = max(contact_depths)
            if max_depth < 0.0001:
                return True
            else:
                return False 
    
        return result.valid