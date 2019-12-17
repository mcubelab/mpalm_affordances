from planning import pushing_planning, grasp_planning, levering_planning, pulling_planning
from helper import util, planning_helper, collisions

import os
from example_config import get_cfg_defaults

from airobot import Robot
from airobot.utils import pb_util, common, arm_util
import pybullet as p
import time
import argparse
import numpy as np

import pickle
import tf.transformations as transformations

from motion_planning.group_planner import GroupPlanner
from motion_planning.motion_planning import MotionPlanner

import numpy as np
import copy
import rospy
import trajectory_msgs
import moveit_commander
import moveit_msgs
from moveit_commander.exception import MoveItCommanderException
from moveit_msgs.srv import GetStateValidity
from tactile_helper.ik import ik_helper

from IPython import embed
from scipy.interpolate import UnivariateSpline
from pykdl_utils.kdl_kinematics import KDLKinematics
from urdf_parser_py.urdf import URDF

import rospy
from trac_ik_python import trac_ik
import threading


class YumiGelslimPybulet():
    def __init__(self, yumi_pb, cfg, exec_thread=True):
        # has an airobot Yumi
        # properly defines which joint is which (wrist, tip, etc.)
        # has a kinematics helper?
        self.cfg = cfg
        self.yumi_pb = yumi_pb

        self.joint_lock = threading.RLock()
        self._both_pos = self.yumi_pb.arm.get_jpos()
        self._single_pos = {}
        self._single_pos['right'] = \
            self.yumi_pb.arm.arm_dict['right'].get_jpos()
        self._single_pos['left'] = \
            self.yumi_pb.arm.arm_dict['left'].get_jpos()

        self.moveit_robot = moveit_commander.RobotCommander()
        self.moveit_scene = moveit_commander.PlanningSceneInterface()
        self.moveit_planner = 'RRTConnectkConfigDefault'

        self.robot_description = '/robot_description'
        self.urdf_string = rospy.get_param(self.robot_description)

        self.mp_left = GroupPlanner(
            'left_arm',
            self.moveit_robot,
            self.moveit_planner,
            self.moveit_scene,
            max_attempts=3,
            planning_time=5.0,
            goal_tol=0.5,
            eef_delta=0.01,
            jump_thresh=10
        )

        self.mp_right = GroupPlanner(
            'right_arm',
            self.moveit_robot,
            self.moveit_planner,
            self.moveit_scene,
            max_attempts=3,
            planning_time=5.0,
            goal_tol=0.5,
            eef_delta=0.01,
            jump_thresh=10
        )

        self.fk_solver_r = KDLKinematics(
            URDF.from_parameter_server(), "yumi_body", "yumi_tip_r")
        self.fk_solver_l = KDLKinematics(
            URDF.from_parameter_server(), "yumi_body", "yumi_tip_l")

        self.num_ik_solver_r = trac_ik.IK(
            'yumi_body', 'yumi_tip_r', urdf_string=self.urdf_string)

        self.num_ik_solver_l = trac_ik.IK(
            'yumi_body', 'yumi_tip_l', urdf_string=self.urdf_string)

        self.ik_pos_tol = 0.001  # 0.001 working well with pulling
        self.ik_ori_tol = 0.01  # 0.01 working well with pulling

        self.execute_thread = threading.Thread(target=self._execute_both)
        self.execute_thread.daemon = True
        if exec_thread:
            self.execute_thread.start()

    def _execute_single(self):
        while True:
            self.joint_lock.acquire()
            self.yumi_pb.arm.set_jpos(self._both_pos, wait=True)
            self.joint_lock.release()

    def _execute_both(self):
        while True:
            self.joint_lock.acquire()
            self.yumi_pb.arm.set_jpos(self._both_pos, wait=True)
            self.joint_lock.release()

    def update_joints(self, pos, arm=None):
        if arm is None:
            self.joint_lock.acquire()
            self._both_pos = pos
            self.joint_lock.release()
        elif arm == 'right':
            both_pos = list(pos) + self.yumi_pb.arm.arm_dict['left'].get_jpos()
            self.joint_lock.acquire()
            self._both_pos = both_pos
            self.joint_lock.release()
        elif arm == 'left':
            both_pos = self.yumi_pb.arm.arm_dict['right'].get_jpos() + list(pos)
            self.joint_lock.acquire()
            self._both_pos = both_pos
            self.joint_lock.release()
        else:
            raise ValueError('Arm not recognized')

    def compute_fk(self, joints, arm='right'):
        if arm == 'right':
            matrix = self.fk_solver_r.forward(joints)
        else:
            matrix = self.fk_solver_l.forward(joints)
        translation = transformations.translation_from_matrix(matrix)
        quat = transformations.quaternion_from_matrix(matrix)
        ee_pose_array = np.hstack((translation, quat))

        ee_pose = util.convert_pose_type(
            ee_pose_array, type_out='PoseStamped', frame_out='yumi_body')
        return ee_pose

    def compute_ik(self, pos, ori, seed, arm='right', solver='trac'):
        if arm != 'right' and arm != 'left':
            arm = 'right'
        if arm == 'right':
            if solver == 'trac':
                sol = self.num_ik_solver_r.get_ik(
                    seed,
                    pos[0],
                    pos[1],
                    pos[2],
                    ori[0],
                    ori[1],
                    ori[2],
                    ori[3],
                    self.ik_pos_tol, self.ik_pos_tol, self.ik_pos_tol,
                    self.ik_ori_tol, self.ik_ori_tol, self.ik_ori_tol
                )
            else:
                sol = self.yumi_pb.arm.compute_ik(
                    pos, ori, arm='right'
                )
        elif arm == 'left':
            if solver == 'trac':
                sol = self.num_ik_solver_l.get_ik(
                    seed,
                    pos[0],
                    pos[1],
                    pos[2],
                    ori[0],
                    ori[1],
                    ori[2],
                    ori[3],
                    self.ik_pos_tol, self.ik_pos_tol, self.ik_pos_tol,
                    self.ik_ori_tol, self.ik_ori_tol, self.ik_ori_tol
                )
            else:
                sol = self.yumi_pb.arm.compute_ik(
                    pos, ori, arm='left'
                )
        return sol

    def unify_arm_trajectories(self, left_arm, right_arm, tip_poses):
        """
        Function to return a right arm and left arm trajectory
        of the same number of points, where the index of the points
        that align with the goal cartesian poses of each arm are the
        same for both trajectories

        Args:
            left_arm (JointTrajectory): left arm joint trajectory from
                left move group after calling compute_cartesian_path
            right_arm (JointTrajectory): right arm joint trajectory from
                right move group after calling compute_cartesian_path
            tip_poses (list): list of desired end effector poses for
                both arms for a particular segment of a primitive plan

        Returns:

        """
        # find the longer trajectory
        long_traj = 'left' if len(left_arm.points) > len(right_arm.points) \
            else 'right'

        # make numpy array of each arm joint trajectory for each comp
        left_arm_joints_np = np.zeros((len(left_arm.points), 7))
        right_arm_joints_np = np.zeros((len(right_arm.points), 7))

        # make numpy array of each arm pose trajectory, based on fk
        left_arm_fk_np = np.zeros((len(left_arm.points), 7))
        right_arm_fk_np = np.zeros((len(right_arm.points), 7))

        for i, point in enumerate(left_arm.points):
            left_arm_joints_np[i, :] = point.positions
            pose = self.compute_fk(point.positions, arm='left')
            left_arm_fk_np[i, :] = util.pose_stamped2list(pose)
        for i, point in enumerate(right_arm.points):
            right_arm_joints_np[i, :] = point.positions
            pose = self.compute_fk(point.positions, arm='right')
            right_arm_fk_np[i, :] = util.pose_stamped2list(pose)

        closest_left_inds = []
        closest_right_inds = []

        # for each tip_pose, find the index in the longer trajectory that
        # most closely matches the pose (using fk)
        for i in range(len(tip_poses)):
            r_waypoint = util.pose_stamped2list(tip_poses[i][1])
            l_waypoint = util.pose_stamped2list(tip_poses[i][0])

            r_pos_diffs = util.pose_difference_np(
                pose=right_arm_fk_np, pose_ref=np.array(r_waypoint))[0]
            l_pos_diffs = util.pose_difference_np(
                pose=left_arm_fk_np, pose_ref=np.array(l_waypoint))[0]
            # r_pos_diffs, r_ori_diffs = util.pose_difference_np(
            #     pose=right_arm_fk_np, pose_ref=np.array(r_waypoint))
            # l_pos_diffs, l_ori_diffs = util.pose_difference_np(
            #     pose=left_arm_fk_np, pose_ref=np.array(l_waypoint))

            r_index = np.argmin(r_pos_diffs)
            l_index = np.argmin(l_pos_diffs)

            # r_index = np.argmin(r_pos_diffs + r_ori_diffs)
            # l_index = np.argmin(l_pos_diffs + l_ori_diffs)

            closest_right_inds.append(r_index)
            closest_left_inds.append(l_index)

        # Create a new trajectory for the shorter trajectory, that is the same
        # length as the longer trajectory.

        if long_traj == 'left':
            new_right = np.zeros((left_arm_joints_np.shape))
            prev_r_ind = 0
            prev_new_ind = 0

            for i, r_ind in enumerate(closest_right_inds):
                # Put the joint values from the short
                # trajectory at the indices corresponding to the path waypoints
                # at the corresponding indices found for the longer trajectory
                new_ind = closest_left_inds[i]
                new_right[new_ind, :] = right_arm_joints_np[r_ind, :]

                # For the missing values in between the joint waypoints,
                # interpolate to fill the trajectory
                # if new_ind - prev_new_ind > -1:
                interp = np.linspace(
                    right_arm_joints_np[prev_r_ind, :],
                    right_arm_joints_np[r_ind],
                    num=new_ind - prev_new_ind)

                new_right[prev_new_ind:new_ind, :] = interp

                prev_r_ind = r_ind
                prev_new_ind = new_ind

            aligned_right_joints = new_right
            aligned_left_joints = left_arm_joints_np
        else:
            new_left = np.zeros((right_arm_joints_np.shape))
            prev_l_ind = 0
            prev_new_ind = 0

            for i, l_ind in enumerate(closest_left_inds):
                new_ind = closest_right_inds[i]
                new_left[new_ind, :] = left_arm_joints_np[l_ind, :]

                interp = np.linspace(
                    left_arm_joints_np[prev_l_ind, :],
                    left_arm_joints_np[l_ind],
                    num=new_ind - prev_new_ind)

                new_left[prev_new_ind:new_ind, :] = interp

                prev_l_ind = l_ind
                prev_new_ind = new_ind

            aligned_right_joints = right_arm_joints_np
            aligned_left_joints = new_left

        # get aligned poses of the end effector as well
        aligned_right_fk = np.zeros((aligned_right_joints.shape[0], 7))
        aligned_left_fk = np.zeros((aligned_left_joints.shape[0], 7))

        for i in range(aligned_right_joints.shape[0]):
            pose_r = self.compute_fk(aligned_right_joints[i, :], arm='right')
            aligned_right_fk[i, :] = util.pose_stamped2list(pose_r)

            pose_l = self.compute_fk(aligned_left_joints[i, :], arm='left')
            aligned_left_fk[i, :] = util.pose_stamped2list(pose_l)

        unified = {}
        unified['right'] = {}
        unified['right']['fk'] = right_arm_fk_np
        unified['right']['joints'] = right_arm_joints_np
        unified['right']['aligned_fk'] = aligned_right_fk
        unified['right']['aligned_joints'] = aligned_right_joints
        unified['right']['inds'] = closest_right_inds

        unified['left'] = {}
        unified['left']['fk'] = left_arm_fk_np
        unified['left']['joints'] = left_arm_joints_np
        unified['left']['aligned_fk'] = aligned_left_fk
        unified['left']['aligned_joints'] = aligned_left_joints
        unified['left']['inds'] = closest_left_inds
        return unified

    def tip_to_wrist(self, tip_poses):
        tip_to_wrist = util.list2pose_stamped(self.cfg.TIP_TO_WRIST_TF, '')
        world_to_world = util.unit_pose()

        wrist_left = util.convert_reference_frame(
            tip_to_wrist,
            world_to_world,
            tip_poses['left'],
            "yumi_body")
        wrist_right = util.convert_reference_frame(
            tip_to_wrist,
            world_to_world,
            tip_poses['right'],
            "yumi_body")

        wrist_poses = {}
        wrist_poses['right'] = wrist_right
        wrist_poses['left'] = wrist_left
        return wrist_poses

    def wrist_to_tip(self, wrist_poses):
        wrist_to_tip = util.list2pose_stamped(self.cfg.WRIST_TO_TIP_TF, '')

        tip_left = util.convert_reference_frame(
            wrist_to_tip,
            util.unit_pose(),
            wrist_poses['left'],
            "yumi_body")
        tip_right = util.convert_reference_frame(
            wrist_to_tip,
            util.unit_pose(),
            wrist_poses['right'],
            "yumi_body")

        tip_poses = {}
        tip_poses['right'] = tip_right
        tip_poses['left'] = tip_left
        return tip_poses

    def is_in_contact(self, object_id):
        r_pts = self.yumi_pb.arm.p.getContactPoints(
            bodyA=self.yumi_pb.arm.robot_id, bodyB=object_id, linkIndexA=12)
        l_pts = self.yumi_pb.arm.p.getContactPoints(
            bodyA=self.yumi_pb.arm.robot_id, bodyB=object_id, linkIndexA=25)

        r_contact_bool = 0 if len(r_pts) == 0 else 1
        l_contact_bool = 0 if len(l_pts) == 0 else 1

        contact_bool = {}
        contact_bool['right'] = r_contact_bool
        contact_bool['left'] = l_contact_bool

        return contact_bool

    def get_jpos(self, arm=None):
        if arm is None:
            jpos = self.yumi_pb.arm.get_jpos()
        elif arm == 'left':
            jpos = self.yumi_pb.arm.arm_dict['left'].get_jpos()
        elif arm == 'right':
            jpos = self.yumi_pb.arm.arm_dict['right'].get_jpos()
        else:
            raise ValueError('Arm not recognized')
        return jpos

    def get_ee_pose(self, arm='right'):
        if arm == 'right':
            ee_pose = self.yumi_pb.arm.get_ee_pose(arm='right')
        elif arm == 'left':
            ee_pose = self.yumi_pb.arm.get_ee_pose(arm='left')
        else:
            raise ValueError('Arm not recognized')
        return ee_pose


class ClosedLoopMacroActions():
    def __init__(self, cfg, robot, object_id, pb_client,
                 config_pkg_path, replan=True):
        # motion planner
        # robot interface? robot set/get functions?
        # ik helper?
        self.cfg = cfg
        self.subgoal_timeout = cfg.SUBGOAL_TIMEOUT
        self.full_plan_timeout = cfg.TIMEOUT
        self.replan = replan

        self.robot = robot
        self.object_id = object_id
        self.pb_client = pb_client

        self.config_pkg_path = config_pkg_path

        self.primitives = ['push', 'pull', 'pivot', 'grasp']
        self.initial_plan = None

        self.goal_pos_tol = 0.005  # 0.003
        self.goal_ori_tol = 0.03  # 0.01

        self.max_ik_iter = 10

    def get_active_arm(self, object_init_pose):
        """
        Returns whether the right arm or left arm
        should be used for pushing, depending on which
        arm the object is closest to

        Args:
            object_init_pose (list): Initial pose of the
                object on the table, in form [x, y, z, x, y, z, w]

        Returns:
            str: 'right' or 'left'
        """
        if object_init_pose[1] > 0:
            active_arm = 'left'
            inactive_arm = 'right'
        else:
            active_arm = 'right'
            inactive_arm = 'left'

        return active_arm, inactive_arm

    def set_replan(self, replan=None):
        if replan is not None and isinstance(replan, bool):
            self.replan = replan

    def get_primitive_plan(self, primitive_name, primitive_args, active_arm):
        # has each primitive function call

        # must pass in the arguments to each primitive
        """
        [summary]

        Args:
            primitive_name ([type]): [description]
            primitive_args ([type]): [description]
            config_path ([type]): [description]
            active_arm ([type]): [description]

        Raises:
            ValueError: [description]

        Returns:
            [type]: [description]
        """
        manipulated_object = primitive_args['object']
        object_pose1_world = primitive_args['object_pose1_world']
        object_pose2_world = primitive_args['object_pose2_world']
        palm_pose_l_object = primitive_args['palm_pose_l_object']
        palm_pose_r_object = primitive_args['palm_pose_r_object']

        if primitive_name == 'push':
            # N = max(primitive_args['N'], 2)

            plan = pushing_planning(
                object=manipulated_object,
                object_pose1_world=object_pose1_world,
                object_pose2_world=object_pose2_world,
                palm_pose_l_object=palm_pose_l_object,
                palm_pose_r_object=palm_pose_r_object,
                arm=active_arm[0])

        elif primitive_name == 'grasp':
            N = max(primitive_args['N'], 4)*2
            plan = grasp_planning(
                object=manipulated_object,
                object_pose1_world=object_pose1_world,
                object_pose2_world=object_pose2_world,
                palm_pose_l_object=palm_pose_l_object,
                palm_pose_r_object=palm_pose_r_object,
                init=primitive_args['init'],
                N=N)

        elif primitive_name == 'pivot':
            gripper_name = self.config_pkg_path + \
                'descriptions/meshes/mpalm/mpalms_all_coarse.stl'
            table_name = self.config_pkg_path + \
                'descriptions/meshes/table/table_top.stl'

            manipulated_object = collisions.CollisionBody(
                self.config_pkg_path +
                'descriptions/meshes/objects/realsense_box_experiments.stl')

            N = max(primitive_args['N'], 2)
            plan = levering_planning(
                object=manipulated_object,
                object_pose1_world=object_pose1_world,
                object_pose2_world=object_pose2_world,
                palm_pose_l_object=palm_pose_l_object,
                palm_pose_r_object=palm_pose_r_object,
                gripper_name=gripper_name,
                table_name=table_name)
            # plan = levering_planning(
            #     object=manipulated_object,
            #     object_pose1_world=object_pose1_world,
            #     object_pose2_world=object_pose2_world,
            #     palm_pose_l_object=palm_pose_l_object,
            #     palm_pose_r_object=palm_pose_r_object,
            #     gripper_name=None,
            #     table_name=None,
            #     N=N)

        elif primitive_name == 'pull':
            N = max(primitive_args['N'], 2)

            plan = pulling_planning(
                object=manipulated_object,
                object_pose1_world=object_pose1_world,
                object_pose2_world=object_pose2_world,
                palm_pose_l_object=palm_pose_l_object,
                palm_pose_r_object=palm_pose_r_object,
                arm=active_arm[0],
                N=N)
        else:
            raise ValueError('Primitive name not recognized')

        return plan

    def greedy_replan(self, primitive_name, object_id, seed,
                      plan_number, frac_done):
        # gets a new primitive plan with args based on robot current state
        object_pos = list(p.getBasePositionAndOrientation(
            object_id, self.pb_client)[0])
        object_ori = list(p.getBasePositionAndOrientation(
            object_id, self.pb_client)[1])

        r_wrist_pos_world = self.robot.get_ee_pose(arm='right')[0].tolist()
        r_wrist_ori_world = self.robot.get_ee_pose(arm='right')[1].tolist()

        l_wrist_pos_world = self.robot.get_ee_pose(arm='left')[0].tolist()
        l_wrist_ori_world = self.robot.get_ee_pose(arm='left')[1].tolist()

        current_wrist_poses = {}
        current_wrist_poses['right'] = util.list2pose_stamped(
            r_wrist_pos_world + r_wrist_ori_world)
        current_wrist_poses['left'] = util.list2pose_stamped(
            l_wrist_pos_world + l_wrist_ori_world
        )

        current_tip_poses = self.robot.wrist_to_tip(current_wrist_poses)

        r_tip_pose_object_frame = util.convert_reference_frame(
            current_tip_poses['right'],
            util.list2pose_stamped(object_pos + object_ori),
            util.unit_pose()
        )

        l_tip_pose_object_frame = util.convert_reference_frame(
            current_tip_poses['left'],
            util.list2pose_stamped(object_pos + object_ori),
            util.unit_pose()
        )

        object_pose_current = util.list2pose_stamped(object_pos + object_ori)

        primitive_args = {}
        primitive_args['object_pose1_world'] = object_pose_current
        primitive_args['object_pose2_world'] = self.object_pose_final
        primitive_args['palm_pose_l_object'] = l_tip_pose_object_frame
        primitive_args['palm_pose_r_object'] = r_tip_pose_object_frame
        primitive_args['object'] = None
        primitive_args['N'] = int(
            len(self.initial_plan[plan_number]['palm_poses_world'])*frac_done)
        primitive_args['init'] = False

        new_plan = self.get_primitive_plan(
            primitive_name,
            primitive_args,
            self.active_arm)

        if primitive_name == 'grasp':
            next_step = 1 if (plan_number == 0 or plan_number == 1) else 14
        else:
            next_step = 1
        new_tip_poses = new_plan[plan_number]['palm_poses_world'][next_step]

        # gets a seed based on the original initial plan
        seed_r = seed['right']
        seed_l = seed['left']

        r_joints = self.robot.compute_ik(
            util.pose_stamped2list(new_tip_poses[1])[:3],
            util.pose_stamped2list(new_tip_poses[1])[3:],
            seed_r,
            arm='right'
        )

        l_joints = self.robot.compute_ik(
            util.pose_stamped2list(new_tip_poses[0])[:3],
            util.pose_stamped2list(new_tip_poses[0])[3:],
            seed_l,
            arm='left'
        )

        joints = {}
        joints['right'] = r_joints
        joints['left'] = l_joints

        return joints, new_plan

    def reach_pose_goal(self, pos, ori, object_id,
                        pos_tol=0.01, ori_tol=0.02):
        """
        Check if end effector reached goal or not. Returns true
        if both position and orientation goals have been reached
        within specified tolerance

        Args:
            pos (list np.ndarray): goal position
            ori (list or np.ndarray): goal orientation. It can be:
                **quaternion** ([qx, qy, qz, qw], shape: :math:`[4]`)
                **rotation matrix** (shape: :math:`[3, 3]`)
                **euler angles** ([roll, pitch, yaw], shape: :math:`[3]`)
            get_func (function): name of the function with which we can get
                the current pose
            pos_tol (float): tolerance of position error
            ori_tol (float): tolerance of orientation error


        Returns:
            bool: If goal pose is reached or not
        """
        if not isinstance(pos, np.ndarray):
            goal_pos = np.array(pos)
        else:
            goal_pos = pos
        if not isinstance(ori, np.ndarray):
            goal_ori = np.array(ori)
        else:
            goal_ori = ori

        if goal_ori.size == 3:
            goal_ori = common.euler2quat(goal_ori)
        elif goal_ori.shape == (3, 3):
            goal_ori = common.rot2quat(goal_ori)
        elif goal_ori.size != 4:
            raise TypeError('Orientation must be in one '
                            'of the following forms:'
                            'rotation matrix, euler angles, or quaternion')
        goal_ori = goal_ori.flatten()
        goal_pos = goal_pos.flatten()

        new_ee_pos = np.array(p.getBasePositionAndOrientation(
            object_id, self.pb_client)[0])
        new_ee_quat = p.getBasePositionAndOrientation(
            object_id, self.pb_client)[1]

        pos_diff = new_ee_pos.flatten() - goal_pos
        pos_error = np.max(np.abs(pos_diff))

        quat_diff = common.quat_multiply(common.quat_inverse(goal_ori),
                                         new_ee_quat)
        rot_similarity = np.abs(quat_diff[3])

        if pos_error < pos_tol and \
                rot_similarity > 1 - ori_tol:
            return True, pos_error, 1-rot_similarity
        else:
            return False, pos_error, 1-rot_similarity

    def execute_single_arm(self, primitive_name, subplan_dict, subplan_goal, subplan_number):
        subplan_tip_poses = subplan_dict['palm_poses_world']

        # setup motion planning request with all the cartesian waypoints
        tip_right = []
        tip_left = []

        r_wrist_pos_world = self.robot.get_ee_pose(arm='right')[0].tolist()
        r_wrist_ori_world = self.robot.get_ee_pose(arm='right')[1].tolist()

        l_wrist_pos_world = self.robot.get_ee_pose(arm='left')[0].tolist()
        l_wrist_ori_world = self.robot.get_ee_pose(arm='left')[1].tolist()

        current_wrist_poses = {}
        current_wrist_poses['right'] = util.list2pose_stamped(
            r_wrist_pos_world + r_wrist_ori_world)
        current_wrist_poses['left'] = util.list2pose_stamped(
            l_wrist_pos_world + l_wrist_ori_world
        )

        current_tip_poses = self.robot.wrist_to_tip(current_wrist_poses)
        current_tip_poses = current_wrist_poses

        # r_tip_pos_world = self.robot.get_ee_pose(arm='right')[0].tolist()
        # r_tip_ori_world = self.robot.get_ee_pose(arm='right')[1].tolist()

        # l_tip_pos_world = self.robot.get_ee_pose(arm='left')[0].tolist()
        # l_tip_ori_world = self.robot.get_ee_pose(arm='left')[1].tolist()

        # current_tip_poses = {}
        # current_tip_poses['right'] = util.list2pose_stamped(
        #     r_tip_pos_world + r_tip_ori_world
        # )
        # current_tip_poses['left'] = util.list2pose_stamped(
        #     l_tip_pos_world + l_tip_ori_world
        # )

        tip_right.append(current_tip_poses['right'].pose)
        tip_left.append(current_tip_poses['left'].pose)

        for i in range(len(subplan_tip_poses)):
            tip_right.append(subplan_tip_poses[i][1].pose)
            tip_left.append(subplan_tip_poses[i][0].pose)

        l_current = self.robot.get_jpos(arm='left')
        r_current = self.robot.get_jpos(arm='right')

        if self.active_arm == 'right':
            traj = self.robot.mp_right.plan_waypoints(
                tip_right,
                force_start=l_current+r_current,
                avoid_collisions=False
            )
        else:
            traj = self.robot.mp_left.plan_waypoints(
                tip_left,
                force_start=l_current+r_current,
                avoid_collisions=False
            )
        # make numpy array of each arm joint trajectory for each comp
        joints_np = np.zeros((len(traj.points), 7))

        # make numpy array of each arm pose trajectory, based on fk
        fk_np = np.zeros((len(traj.points), 7))

        for i, point in enumerate(traj.points):
            joints_np[i, :] = point.positions
            pose = self.robot.compute_fk(
                point.positions,
                arm=self.active_arm)
            fk_np[i, :] = util.pose_stamped2list(pose)

        reached_goal, pos_err_total, ori_err_total = self.reach_pose_goal(
            subplan_goal[:3],
            subplan_goal[3:],
            self.object_id,
            pos_tol=self.goal_pos_tol, ori_tol=self.goal_ori_tol)

        pos_err = pos_err_total
        ori_err = ori_err_total

        start_time = time.time()
        timed_out = False

        made_contact = False
        while not reached_goal:
            pose_ref = util.pose_stamped2list(
                self.robot.compute_fk(
                    joints=self.robot.get_jpos(arm=self.active_arm),
                    arm=self.active_arm))
            diffs = util.pose_difference_np(
                pose=fk_np,
                pose_ref=pose_ref)[0]

            seed_ind = min(np.argmin(diffs), joints_np.shape[0]-2)

            seed = {}
            seed[self.active_arm] = joints_np[seed_ind, :]
            seed[self.inactive_arm] = \
                self.robot.get_jpos(arm=self.inactive_arm)

            if self.replan and made_contact:
                ik_iter = 0
                ik_sol_found = False
                while not ik_sol_found:
                    joints_execute = self.greedy_replan(
                        primitive_name=primitive_name,
                        object_id=self.object_id,
                        seed=seed,
                        plan_number=subplan_number,
                        frac_done=pos_err/pos_err_total)[0][self.active_arm]
                    if joints_execute is not None:
                        ik_sol_found = True
                    ik_iter += 1

                    if ik_iter > self.max_ik_iter:
                        raise ValueError('IK solution not found!')
            else:
                joints_execute = joints_np[seed_ind+1, :].tolist()

            self.robot.update_joints(joints_execute, arm=self.active_arm)

            reached_goal, pos_err, ori_err = self.reach_pose_goal(
                subplan_goal[:3],
                subplan_goal[3:],
                self.object_id,
                pos_tol=self.goal_pos_tol, ori_tol=self.goal_ori_tol)

            timed_out = time.time() - start_time > self.subgoal_timeout
            if timed_out:
                print("TIMED OUT!")
                break
            time.sleep(0.001)
            if not made_contact:
                made_contact = self.robot.is_in_contact(self.object_id)[self.active_arm]
        return reached_goal, pos_err, ori_err

    def execute_two_arm(self, primitive_name, subplan_dict, subplan_goal, subplan_number):
        subplan_tip_poses = subplan_dict['palm_poses_world']

        # setup motion planning request with all the cartesian waypoints
        tip_right = []
        tip_left = []

        for i in range(len(subplan_tip_poses)):
            tip_right.append(subplan_tip_poses[i][1].pose)
            tip_left.append(subplan_tip_poses[i][0].pose)

        l_current = self.robot.get_jpos(arm='left')
        r_current = self.robot.get_jpos(arm='right')

        traj_right = self.robot.mp_right.plan_waypoints(
            tip_right,
            force_start=l_current+r_current,
            avoid_collisions=False
        )

        traj_left = self.robot.mp_left.plan_waypoints(
            tip_left,
            force_start=l_current+r_current,
            avoid_collisions=False
        )

        unified = self.robot.unify_arm_trajectories(
            traj_left,
            traj_right,
            subplan_tip_poses)

        aligned_left = unified['left']['aligned_joints']
        aligned_right = unified['right']['aligned_joints']

        if aligned_left.shape != aligned_right.shape:
            raise ValueError('Could not aligned joint trajectories')

        reached_goal, pos_err_total, ori_err_total = self.reach_pose_goal(
            subplan_goal[:3],
            subplan_goal[3:],
            self.object_id,
            pos_tol=self.goal_pos_tol, ori_tol=self.goal_ori_tol)

        pos_err = pos_err_total
        ori_err = ori_err_total

        timed_out = False
        start_time = time.time()

        while not reached_goal:
            # check if replanning or not

            # find closest point in original motion plan
            pose_ref_r = util.pose_stamped2list(
                self.robot.compute_fk(
                    joints=self.robot.get_jpos(arm='right'),
                    arm='right'))
            diffs_r = util.pose_difference_np(
                pose=unified['right']['aligned_fk'][:],
                pose_ref=pose_ref_r)

            pose_ref_l = util.pose_stamped2list(
                self.robot.compute_fk(
                    joints=self.robot.get_jpos('left'),
                    arm='left'))
            diffs_l = util.pose_difference_np(
                pose=unified['left']['aligned_fk'][:],
                pose_ref=pose_ref_l)

            seed_ind_r = min(np.argmin(diffs_r[0]), aligned_right.shape[0]-2)
            seed_ind_l = min(np.argmin(diffs_l[0]), aligned_left.shape[0]-2)
            if not self.replan and (seed_ind_l == aligned_left.shape[0]-2 or \
                    seed_ind_r == aligned_right.shape[0]-2):
                reached_goal = True

            seed = {}
            seed['right'] = aligned_right[:, :][seed_ind_r, :]
            seed['left'] = aligned_left[:, :][seed_ind_l, :]

            # if self.replan and subplan_number == 1:
            both_contact = self.robot.is_in_contact(self.object_id)['right'] and \
                self.robot.is_in_contact(self.object_id)['left']
            if self.replan and both_contact:
                print("replanning!")
                ik_iter = 0
                ik_sol_found = False
                while not ik_sol_found:
                    joints_execute = self.greedy_replan(
                        primitive_name=primitive_name,
                        object_id=self.object_id,
                        seed=seed,
                        plan_number=subplan_number,
                        frac_done=pos_err/pos_err_total)[0]
                    if joints_execute is not None:
                        ik_sol_found = True
                    ik_iter += 1
                    if ik_iter > self.max_ik_iter:
                        raise ValueError('IK solution not found!')
                both_joints = joints_execute['right'] + joints_execute['left']
            else:
                # move to the next point w.r.t. closest point
                r_pos = aligned_right[:, :][seed_ind_r+1, :].tolist()
                l_pos = aligned_left[:, :][seed_ind_r+1, :].tolist()
                both_joints = r_pos + l_pos

            # set joint target
            self.robot.update_joints(both_joints)

            # see how far we are from the goal
            reached_goal, pos_err, ori_err = self.reach_pose_goal(
                subplan_goal[:3],
                subplan_goal[3:],
                self.object_id,
                pos_tol=self.goal_pos_tol, ori_tol=self.goal_ori_tol
            )
            timed_out = time.time() - start_time > self.subgoal_timeout
            if timed_out:
                print("Timed out!")
                break
            both_contact = self.robot.is_in_contact(self.object_id)['right'] and \
                self.robot.is_in_contact(self.object_id)['left']
            time.sleep(0.01)
        return reached_goal, pos_err, ori_err

    def execute(self, primitive_name, execute_args):
        if primitive_name not in self.primitives:
            raise ValueError('Primitive not recognized')

        self.object_pose_init = \
            util.pose_stamped2list(execute_args['object_pose1_world'])
        self.object_pose_final = \
            execute_args['object_pose2_world']

        self.active_arm, self.inactive_arm = \
            self.get_active_arm(self.object_pose_init)

        if primitive_name == 'pivot' or primitive_name == 'grasp':
            two_arm = True
        else:
            two_arm = False

        self.initial_plan = self.get_primitive_plan(
            primitive_name,
            execute_args,
            self.active_arm
        )

        subplan_number = 0
        done = False
        start = time.time()
        while not done:
            full_execute_time = time.time() - start
            if full_execute_time > self.full_plan_timeout:
                done = True

            # start going through subplans from initial plan
            subplan_dict = self.initial_plan[subplan_number]

            # get intermediate_goal
            subplan_goal = util.pose_stamped2list(
                subplan_dict['object_poses_world'][-1]
            )

            if two_arm:
                success, pos_err, ori_err = self.execute_two_arm(
                    primitive_name,
                    subplan_dict,
                    subplan_goal,
                    subplan_number
                )
            else:
                success, pos_err, ori_err = self.execute_single_arm(
                    primitive_name,
                    subplan_dict,
                    subplan_goal,
                    subplan_number
                )

            subplan_number += 1
            if subplan_number > len(self.initial_plan) - 1:
                done = True
        return success, pos_err, ori_err


def visualize_goal_state(object_id, goal_pose, pb_client):
    while True:
        p.resetBasePositionAndOrientation(
            object_id,
            [goal_pose[0], goal_pose[1], goal_pose[2]+0.01],
            goal_pose[3:],
            physicsClientId=pb_client)
        time.sleep(0.01)


def main(args):
    cfg_file = os.path.join(args.example_config_path, args.primitive) + ".yaml"
    cfg = get_cfg_defaults()
    cfg.merge_from_file(cfg_file)
    cfg.freeze()

    rospy.init_node('MacroActions')

    # setup yumi
    yumi_ar = Robot('yumi',
                    pb=True,
                    arm_cfg={'render': True, 'self_collision': False})
    yumi_ar.arm.set_jpos(cfg.RIGHT_INIT + cfg.LEFT_INIT)

    gel_id = 12
    # embed()

    # p.changeDynamics(
    #     yumi_ar.arm.robot_id,
    #     gel_id,
    #     rollingFriction=args.rolling
    # )

    alpha = 0.01
    K = 500
    # p.changeDynamics(
    #     yumi_ar.arm.robot_id,
    #     gel_id,
    #     contactStiffness=K,
    #     contactDamping=alpha*K
    # )    

    # 0.9999 working well...?
    # p.changeDynamics(
    #     yumi_ar.arm.robot_id,
    #     gel_id,
    #     restitution=0.9999,
    # )

    p.changeDynamics(
        yumi_ar.arm.robot_id,
        gel_id,
        restitution=0.7,
        contactStiffness=K,
        contactDamping=alpha*K
    )

    #
    # table_id = 27
    # p.changeDynamics(
    #     yumi_ar.arm.robot_id,
    #     table_id,
    #     contactStiffness=500,
    #     contactDamping=1)    

    # setup yumi_gs
    yumi_gs = YumiGelslimPybulet(yumi_ar, cfg)

    if args.object:
        box_id = pb_util.load_urdf(
            args.config_package_path +
            'descriptions/urdf/'+args.object_name+'.urdf',
            cfg.OBJECT_INIT[0:3],
            cfg.OBJECT_INIT[3:]
        )

    # setup macro_planner
    action_planner = ClosedLoopMacroActions(
        cfg,
        yumi_gs,
        box_id,
        pb_util.PB_CLIENT,
        args.config_package_path,
        replan=args.replan
    )

    manipulated_object = None
    object_pose1_world = util.list2pose_stamped(cfg.OBJECT_INIT)
    object_pose2_world = util.list2pose_stamped(cfg.OBJECT_FINAL)
    palm_pose_l_object = util.list2pose_stamped(cfg.PALM_LEFT)
    palm_pose_r_object = util.list2pose_stamped(cfg.PALM_RIGHT)

    example_args = {}
    example_args['object_pose1_world'] = object_pose1_world
    example_args['object_pose2_world'] = object_pose2_world
    example_args['palm_pose_l_object'] = palm_pose_l_object
    example_args['palm_pose_r_object'] = palm_pose_r_object
    example_args['object'] = manipulated_object
    example_args['N'] = 60  # 60
    example_args['init'] = True

    primitive_name = args.primitive

    trans_box_id = pb_util.load_urdf(
        args.config_package_path +
        'descriptions/urdf/'+args.object_name+'_trans.urdf',
        cfg.OBJECT_FINAL[0:3],
        cfg.OBJECT_FINAL[3:]
    )  
    visualize_goal_thread = threading.Thread(
        target=visualize_goal_state,
        args=(trans_box_id, cfg.OBJECT_FINAL, action_planner.pb_client))
    visualize_goal_thread.daemon = True
    visualize_goal_thread.start()

    result = action_planner.execute(primitive_name, example_args)

    embed()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--config_package_path',
        type=str,
        default='/root/catkin_ws/src/config/')

    parser.add_argument(
        '--example_config_path',
        type=str,
        default='config')

    parser.add_argument(
        '--primitive',
        type=str,
        default='push',
        help='which primitive to plan')

    parser.add_argument(
        '--simulate',
        type=int,
        default=1)

    parser.add_argument(
        '-o',
        '--object',
        action='store_true')

    parser.add_argument(
        '-re',
        '--replan',
        action='store_true',
        default=False)

    parser.add_argument(
        '--object_name',
        type=str,
        default='realsense_box')

    parser.add_argument(
        '-r', '--rolling',
        type=float,
        default=0.0,
        help='rolling friction value for changeDynamics in pybullet'
    )

    args = parser.parse_args()
    main(args)
