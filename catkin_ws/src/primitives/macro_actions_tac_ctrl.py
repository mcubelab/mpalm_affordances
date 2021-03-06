import os
import time
import argparse
import numpy as np
import copy
import threading
import trimesh
from IPython import embed

import rospy
from scipy.interpolate import UnivariateSpline
from pykdl_utils.kdl_kinematics import KDLKinematics
from urdf_parser_py.urdf import URDF
from trac_ik_python import trac_ik
from geometry_msgs.msg import PoseStamped
import tf.transformations as transformations
import moveit_commander

from planning import pushing_planning, grasp_planning
from planning import levering_planning, pulling_planning
from helper import util, collisions
from motion_planning.group_planner import GroupPlanner

import pybullet as p
from airobot import Robot
from airobot.utils import pb_util, common
# from airobot.utils import pb_util, common
# from airobot.utils.pb_util import step_simulation

# from example_config_cfg import get_cfg_defaults
from closed_loop_experiments_cfg import get_cfg_defaults

from tactile_controller.tactile_model import (
    initialize_levering_tactile_setup, initialize_grasping_tactile_setup,
    initialize_pulling_tactile_setup, TactileControl)


class YumiGelslimPybulet(object):
    """
    Class for interfacing with Yumi in PyBullet
    with external motion planning, inverse kinematics,
    and forward kinematics, along with other helpers
    """
    def __init__(self, yumi_pb, cfg, exec_thread=True, sim_step_repeat=10):
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
        self.cfg = cfg
        self.yumi_pb = yumi_pb
        self.sim_step_repeat = sim_step_repeat

        self.joint_lock = threading.RLock()
        self.sleep_lock = threading.RLock()
        self._both_pos = self.yumi_pb.arm.get_jpos()
        self._single_pos = {}
        self._single_pos['right'] = \
            self.yumi_pb.arm.arms['right'].get_jpos()
        self._single_pos['left'] = \
            self.yumi_pb.arm.arms['left'].get_jpos()

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
        self._sim_step_sleep = 0.01
        # self._sim_step_sleep = 0.004
        self._max_velocity = None
        self._max_velocities = None
        self._both_vel = np.asarray([0.0]*self.yumi_pb.arm.dual_arm_dof)
        self._vel_signs = np.asarray([True]*self.yumi_pb.arm.dual_arm_dof)
        if exec_thread:
            self.execute_thread.start()
            self.step_sim_mode = False
        else:
            self.step_sim_mode = True

    def set_sim_sleep(self, sleep_t, max_velocity=None):
        self.sleep_lock.acquire()
        self._sim_step_sleep = sleep_t
        self._max_velocity = max_velocity
        self.sleep_lock.release()

    def _execute_single(self):
        """
        Background thread for controlling a single arm
        """
        while True:
            self.joint_lock.acquire()
            self.yumi_pb.arm.set_jpos(self._both_pos, wait=True)
            self.joint_lock.release()

    def _execute_both(self):
        """
        Background thread for controlling both arms
        """
        while True:
            self.joint_lock.acquire()
            # self.yumi_pb.arm.set_jpos(
            #     self._both_pos,
            #     wait=False,
            #     max_velocity=self._max_velocities)
            # self.yumi_pb.arm.set_jpos(
            #     self._both_pos,
            #     wait=False,
            #     velocity_gain=0.1)
            self.yumi_pb.arm.set_jpos(
                self._both_pos,
                wait=False)
            self.yumi_pb.pb_client.stepSimulation()
            self.joint_lock.release()

            self.sleep_lock.acquire()
            time.sleep(self._sim_step_sleep)
            self.sleep_lock.release()

    def update_joints(self, pos, arm=None, prev=None):
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
        # if prev is not None and self._max_velocity is not None:
        #     self._both_vel = np.asarray(self._both_pos) - np.asarray(prev)
        #     neg_inds = np.where(self._both_vel < 0.0)[0]
        #     self._vel_signs = np.ones((self.yumi_pb.arm.dual_arm_dof,))
        #     self._vel_signs[neg_inds] = -1.0
        #     self._max_velocities = self._vel_signs*self._max_velocity
        # else:
        #     self._max_velocities = None
        # if self.step_sim_mode:
        #     for _ in range(self.sim_step_repeat):
        #         # step_simulation()
        #         self.yumi_pb.pb_client.stepSimulation()

    def compute_fk(self, joints, arm='right'):
        """
        Forward kinematics calculation.

        Args:
            joints (list): Joint configuration, should be len() =
                DOF of a single arm (7 for Yumi)
            arm (str, optional): Which arm to compute FK for.
                Defaults to 'right'.

        Returns:
            PoseStamped: End effector pose corresponding to
                the FK solution for the input joint configuation
        """
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
        """
        Inverse kinematics calcuation

        Args:
            pos (list): Desired end effector position [x, y, z]
            ori (list): Desired end effector orientation (quaternion),
                [x, y, z, w]
            seed (list): Initial solution guess to IK calculation.
                Returned solution will be near the seed.
            arm (str, optional): Which arm to compute IK for.
                Defaults to 'right'.
            solver (str, optional): Which IK solver to use.
                Defaults to 'trac'.

        Returns:
            list: Configuration space solution corresponding to the
                desired end effector pose. len() = DOF of single arm
        """
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
            dict: Dictionary of combined trajectories in different formats.
                Keys for each arm, 'right' and 'left', which each are
                themselves dictionary. Deeper keys ---
                'fk': Cartesian path numpy array, unaligned
                'joints': C-space path numpy array, unaligned
                'aligned_fk': Cartesian path numpy array, aligned
                'aligned_joints': C-space path numpy array, aligned
                'closest_inds': Indices of each original path corresponding
                    to the closest value in the other arms trajectory
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
        """
        Transform a pose from the Yumi Gelslim tip to the
        wrist joint

        Args:
            tip_poses (dict): Dictionary of PoseStamped values
                corresponding to each arm, keyed by 'right' and
                'left'

        Returns:
            dict: Keyed by 'right' and 'left', values are PoseStamped
        """
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
        """
        Transform a pose from the Yumi wrist joint to the
        Gelslim tip to the

        Args:
            wrist_poses (dict): Dictionary of PoseStamped values
                corresponding to each arm, keyed by 'right' and
                'left'

        Returns:
            dict: Keyed by 'right' and 'left', values are PoseStamped
        """
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
        """
        Checks whether or not robot is in contact with a
        particular object

        Args:
            object_id (int): Pybullet object ID of object contact
                is checked with

        Returns:
            dict: Keyed by 'right' and 'left', values are bools.
                True means arm 'right/left' is in contact, else False
        """
        # r_pts = p.getContactPoints(
        #     bodyA=self.yumi_pb.arm.robot_id, bodyB=object_id, linkIndexA=12, physicsClientId=pb_util.PB_CLIENT)
        # l_pts = p.getContactPoints(
        #     bodyA=self.yumi_pb.arm.robot_id, bodyB=object_id, linkIndexA=25, physicsClientId=pb_util.PB_CLIENT)
        r_pts = p.getContactPoints(
            bodyA=self.yumi_pb.arm.robot_id, bodyB=object_id, linkIndexA=12, physicsClientId=self.yumi_pb.pb_client.get_client_id())
        l_pts = p.getContactPoints(
            bodyA=self.yumi_pb.arm.robot_id, bodyB=object_id, linkIndexA=25, physicsClientId=self.yumi_pb.pb_client.get_client_id())

        r_contact_bool = 0 if len(r_pts) == 0 else 1
        l_contact_bool = 0 if len(l_pts) == 0 else 1

        contact_bool = {}
        contact_bool['right'] = r_contact_bool
        contact_bool['left'] = l_contact_bool

        return contact_bool

    def get_jpos(self, arm=None):
        """
        Getter function for getting the robot's joint positions

        Args:
            arm (str, optional): Which arm to get the position for.
                Defaults to None, if None will return position of
                both arms.

        Returns:
            list: Joint positions, len() = DOF of single arm
        """
        if arm is None:
            jpos = self.yumi_pb.arm.get_jpos()
        elif arm == 'left':
            jpos = self.yumi_pb.arm.arms['left'].get_jpos()
        elif arm == 'right':
            jpos = self.yumi_pb.arm.arms['right'].get_jpos()
        else:
            raise ValueError('Arm not recognized')
        return jpos

    def get_ee_pose(self, arm='right'):
        """
        Getter function for getting the robot end effector pose

        Args:
            arm (str, optional): Which arm to get the EE pose for.
                Defaults to 'right'.

        Returns:
            4-element tuple containing
            - np.ndarray: x, y, z position of the EE (shape: :math:`[3,]`)
            - np.ndarray: quaternion representation of the
              EE orientation (shape: :math:`[4,]`)
            - np.ndarray: rotation matrix representation of the
              EE orientation (shape: :math:`[3, 3]`)
            - np.ndarray: euler angle representation of the
              EE orientation (roll, pitch, yaw with
              static reference frame) (shape: :math:`[3,]`)
        """
        if arm == 'right':
            ee_pose = self.yumi_pb.arm.get_ee_pose(arm='right')
        elif arm == 'left':
            ee_pose = self.yumi_pb.arm.get_ee_pose(arm='left')
        else:
            raise ValueError('Arm not recognized')
        return ee_pose


class ClosedLoopMacroActions():
    """
    Class for interfacing with a set of reactive motion primitives
    """
    def __init__(self, cfg, robot, object_id, pb_client,
                 config_pkg_path, object_mesh_file=None, replan=True,
                 contact_face=None, perturb=False):
        """
        Constructor for MacroActions class. Sets up
        internal interface to the robot, and settings for the
        primitive planners and executors

        Args:
            cfg (YACS CfgNode): Configuration
            robot (YumiGelslimPybullet): Robot class that includes
                pybullet interface, IK/FK helpers, motion planning, etc.
            object_id (int): Pybullet object id that is being interacted with
            pb_client (int): PyBullet client id for connecting to physics
                simulation
            config_pkg_path (str): Absolute file path to ROS config package
            replan (bool, optional): Whether or not primitives should be
                executed with the object state controller running (MPC style
                replanning). Defaults to True.
        """
        self.cfg = cfg
        self.subgoal_timeout = cfg.SUBGOAL_TIMEOUT
        self.full_plan_timeout = cfg.TIMEOUT
        self.replan = replan

        self.robot = robot
        self.pb_client = pb_client

        self.config_pkg_path = config_pkg_path

        self.primitives = ['push', 'pull', 'pivot', 'grasp']
        self.initial_plan = None

        self.goal_pos_tol = 0.003  # 0.003
        self.goal_ori_tol = 0.01  # 0.01

        self.max_ik_iter = 20

        # self.object_id = object_id
        # self.object_mesh_file = object_mesh_file
        # self.mesh = trimesh.load_mesh(self.object_mesh_file)
        # self.mesh.apply_translation(-self.mesh.center_mass)
        # self.mesh_world = copy.deepcopy(self.mesh)

        self.update_object(object_id, object_mesh_file)

        self.contact_face = contact_face

        self.kp = 1
        self.kd = 0.001

        self.default_active_arm = 'right'
        self.default_inactive_arm = 'left'

        self.state_lock = threading.Lock()
        self.tactile_state_estimator = threading.Thread(target=self.palm_object_state_estimator)
        self.tactile_state_estimator.daemon = True

        self.perturb = perturb
        self.tactile_thread_started = False

        # if primitive_name
        # self._object, self.contact_dict, self.optimization_equilibrium_parameters, self.optimization_control_parameters = initialize_levering_tactile_setup()
        # self.tactile_control = TactileControl(self._object, self.contact_dict, self.optimization_equilibrium_parameters, self.optimization_control_parameters)

        # self.table_object_contact_list = []
        # self.right_object_contact_list = []
        # self.left_object_contact_list = []
        # self.vertex_positions = []

    def get_active_arm(self, object_init_pose, use_default=True):
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
        if use_default:
            active_arm = self.default_active_arm
            inactive_arm = self.default_inactive_arm
        else:
            if object_init_pose[1] > 0:
                active_arm = 'left'
                inactive_arm = 'right'
            else:
                active_arm = 'right'
                inactive_arm = 'left'

        return active_arm, inactive_arm

    def get_nominal_palms(self, execute_args):
        """
        Get the original world frame and object frame
        palm poses, for enforcing target positions in
        certain axis to prevent slipping during replanning
        (pseudo tactile controller)

        Args:
            execute_args (dict): Dictionary with initial primitive arguments

        Returns:
            dict: right and left palm poses, in both world and initial object frame
        """
        nominal_palm_pose_r_obj = execute_args['palm_pose_r_object']
        nominal_palm_pose_l_obj = execute_args['palm_pose_l_object']
        nominal_palm_pose_r_world = util.convert_reference_frame(
            nominal_palm_pose_r_obj,
            util.unit_pose(),
            execute_args['object_pose1_world']
        )
        nominal_palm_pose_l_world = util.convert_reference_frame(
            nominal_palm_pose_l_obj,
            util.unit_pose(),
            execute_args['object_pose1_world']
        )
        palms = {}
        palms['right'] = {}
        palms['left'] = {}
        palms['right']['obj'] = nominal_palm_pose_r_obj
        palms['right']['world'] = nominal_palm_pose_r_world
        palms['left']['obj'] = nominal_palm_pose_l_obj
        palms['left']['world'] = nominal_palm_pose_l_world

        return palms

    def update_object(self, obj_id, mesh_file):
        """
        Update the internal variables associated with the object
        in the environment, so that contacts can be checked
        and the mesh can be used

        Args:
            obj_id (int): PyBullet object id
            mesh_file (str): Path to .stl file of the object
        """
        self.object_id = obj_id
        self.object_mesh_file = mesh_file
        self.mesh = trimesh.load_mesh(self.object_mesh_file)
        self.mesh.apply_translation(-self.mesh.center_mass)
        # self.mesh.apply_scale(0.001)
        self.mesh_world = copy.deepcopy(self.mesh)

    def transform_mesh_world(self, pose=None):
        """
        Interal method to transform the object mesh coordinates
        to the world frame, based on where it is in the environment
        """
        self.mesh_world = copy.deepcopy(self.mesh)
        if pose is None:
            obj_pos_world = list(p.getBasePositionAndOrientation(
                self.object_id, self.pb_client)[0])
            obj_ori_world = list(p.getBasePositionAndOrientation(
                self.object_id, self.pb_client)[1])
        else:
            obj_pos_world = pose[:3]
            obj_ori_world = pose[3:]
        obj_ori_mat = common.quat2rot(obj_ori_world)
        h_trans = np.zeros((4, 4))
        h_trans[:3, :3] = obj_ori_mat
        h_trans[:-1, -1] = obj_pos_world
        h_trans[-1, -1] = 1
        self.mesh_world.apply_transform(h_trans)

    def get_contact_face_normal(self):
        """
        Gets the updated world frame surface normal of the face which is in
        contact with the palm
        """
        if self.contact_face is not None:
            self.transform_mesh_world()
            face_normal = self.mesh_world.face_normals[self.contact_face]
            return face_normal
        else:
            return None

    def get_palm_y_normals(self, palm_poses=None, *args):
        """
        Gets the updated world frame normal direction of the palms
        """
        normal_y = util.list2pose_stamped([0, 1, 0, 0, 0, 0, 1])

        normal_y_poses_world = {}
        wrist_poses = {}

        for arm in ['right', 'left']:
            if palm_poses is None:
                wrist_pos_world = self.robot.get_ee_pose(arm=arm)[0].tolist()
                wrist_ori_world = self.robot.get_ee_pose(arm=arm)[1].tolist()

                wrist_poses[arm] = util.list2pose_stamped(wrist_pos_world + wrist_ori_world)
            else:
                wrist_poses[arm] = palm_poses[arm]

        tip_poses = self.robot.wrist_to_tip(wrist_poses)

        normal_y_poses_world['right'] = util.transform_pose(normal_y, tip_poses['right'])
        normal_y_poses_world['left'] = util.transform_pose(normal_y, tip_poses['left'])

        return normal_y_poses_world

    def get_palm_z_normals(self, palm_poses=None, *args):
        """
        Gets the updated world frame normal direction of the palms
        """
        normal_z = util.list2pose_stamped([0, 0, 1, 0, 0, 0, 1])

        normal_z_poses_world = {}
        wrist_poses = {}

        for arm in ['right', 'left']:
            if palm_poses is None:
                wrist_pos_world = self.robot.get_ee_pose(arm=arm)[0].tolist()
                wrist_ori_world = self.robot.get_ee_pose(arm=arm)[1].tolist()

                wrist_poses[arm] = util.list2pose_stamped(wrist_pos_world + wrist_ori_world)
            else:
                wrist_poses[arm] = palm_poses[arm]

        tip_poses = self.robot.wrist_to_tip(wrist_poses)

        normal_z_poses_world['right'] = util.transform_pose(normal_z, tip_poses['right'])
        normal_z_poses_world['left'] = util.transform_pose(normal_z, tip_poses['left'])

        return normal_z_poses_world

    def get_current_tip_poses(self):

        wrist_poses = {}

        for arm in ['right', 'left']:
            wrist_pos_world = self.robot.get_ee_pose(arm=arm)[0].tolist()
            wrist_ori_world = self.robot.get_ee_pose(arm=arm)[1].tolist()

            wrist_poses[arm] = util.list2pose_stamped(wrist_pos_world + wrist_ori_world)

        tip_poses = self.robot.wrist_to_tip(wrist_poses)

        return tip_poses

    def get_primitive_plan(self, primitive_name, primitive_args, active_arm):
        """
        Wrapper function for getting the nominal plan for each primitive action

        Args:
            primitive_name (str): Which primitive to get plan for. Must be one of
                the primitive names in self.primitives
            primitive_args (dict): Contains all the arguments necessary to call
                the primitive planner, including initial object pose, goal
                object pose, palm contact poses in the object frame, number of
                waypoints to use, and others.
            active_arm (str): 'right' or 'left', which arm to return plan for,
                if using single arm primitive

        Returns:
            list: (list of dict with keys)
                palm_poses_r_world (list of util.PoseStamped): Trajectory of
                    right palm poses in world frame
                palm_poses_l_world (list of util.PoseStamped): Trajectory of
                    left palm poses in world frame
                object_poses_world (util.PoseStamped): Trajectory of object
                    poses in world frame
                primitive (util.PoseStamped): Name of primitive
                    (i.e., 'grasping') name (util.PoseStamped): Name of plan
                t (util.PoseStamped): list of timestamps associated with each
                    pose
                N (util.PoseStamped): Number of keypoints in the plan
                    (i.e., len(plan_dict['t'])
        """
        manipulated_object = primitive_args['object']
        object_pose1_world = primitive_args['object_pose1_world']
        object_pose2_world = primitive_args['object_pose2_world']
        palm_pose_l_object = primitive_args['palm_pose_l_object']
        palm_pose_r_object = primitive_args['palm_pose_r_object']
        table_face = primitive_args['table_face']

        if primitive_name == 'push':
            # N = max(primitive_args['N'], 2)

            plan = pushing_planning(
                object=manipulated_object,
                object_pose1_world=object_pose1_world,
                object_pose2_world=object_pose2_world,
                palm_pose_l_object=palm_pose_l_object,
                palm_pose_r_object=palm_pose_r_object,
                arm=active_arm[0],
                table_face=table_face)

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
                table_name=table_name,
                N=N)

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
                      plan_number, frac_done, simulate=True):
        """
        Replanning function, which functions as the object state
        controller in an MPC style. Compute the current state of the
        world, including object state and palm poses in the object frame,
        and calls the primitive planners with the updated arguments based
        on wherever it is during execution.

        Args:
            primitive_name (str): Which primitive is being planned for
            object_id (int): PyBullet object id of object being manipulated
            seed (list): IK solution seed. This should come from the nominal
                initial primitive plan, which is expected not to have
                singularities or large C-space jumps
            plan_number (int): Which subplan we are on during execution. Will
                only be nonzero for multistep plans, i.e. grasping, which
                first lift the object (plan_number=0),
                reorient it in some way (plan_number=1), and then put it
                back down (plan_number=2).
            frac_done (float): Decimal value of what fraction of the way to
                the goal pose the object is currently at.

        Returns:
            2-element tupe containing:
            - dict: Joint values to move to at the next time step, keyed by
              'right' and 'left'
            - list: New plan from the replanning call
              (from self.get_primitive_plan)
        """
        # gets a new primitive plan with args based on robot current state
        object_pos = list(p.getBasePositionAndOrientation(
            object_id, self.pb_client)[0])
        object_ori = list(p.getBasePositionAndOrientation(
            object_id, self.pb_client)[1])
        object_pose_current = util.list2pose_stamped(object_pos + object_ori)

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
        if primitive_name == 'pull':
            current_z_normals = self.get_palm_z_normals(None, current=True)
            current_tip_poses['right'].pose.position.z = self.nominal_palms['right']['world'].pose.position.z - 0.002
            # current_tip_poses['left'].pose.position.z = self.nominal_palms['left']['world'].pose.position.z - 0.002
            self.state_lock.acquire()
            ds = self._delta_spring
            self.state_lock.release()
            # current_tip_poses['right'].pose.position.z = self.nominal_palms['right']['world'].pose.position.z + self._delta_spring
            # current_tip_poses['left'].pose.position.z = self.nominal_palms['left']['world'].pose.position.z + self._delta_spring
            for arm in ['right', 'left']:
                # current_tip_poses['left'].pose.position.z = self.nominal_palms['left']['world'].pose.position.z + self._delta_spring
                current_pos = np.asarray(util.pose_stamped2list(current_tip_poses[arm])[:3])

                y_vec = [0, 0, 1]
                z_vec = np.asarray(util.pose_stamped2list(current_z_normals[arm])[:3]) - current_pos
                x_vec = np.cross(y_vec, z_vec)
                current_tip_poses[arm] = util.pose_from_vectors(
                    x_vec, y_vec, z_vec, current_pos
                )
                # current_tip_poses[arm].pose.position.z = self.nominal_palms['right']['world'].pose.position.z + self._delta_spring
                current_tip_poses['right'].pose.position.z += ds

        if primitive_name == 'push':
            _, _, vectors, _, _ = self.pushing_normal_alignment(show=False)
            x_vec, y_vec, z_vec = vectors[0], vectors[1], vectors[2]

            for arm in ['right', 'left']:
                current_pos = util.pose_stamped2list(current_tip_poses[arm])[:3]
                nominal_pos = util.pose_stamped2list(
                    util.convert_reference_frame(
                        util.list2pose_stamped(self.cfg.PALM_RIGHT),
                        util.unit_pose(),
                        object_pose_current))[:3]
                self.state_lock.acquire()
                self.transform_mesh_world()
                try:
                    nearest_pos = self.mesh_world.nearest.on_surface(np.asarray(current_pos)[np.newaxis, :])[0][0]
                except ValueError as e:
                    nearest_pos = current_pos
                self.state_lock.release()
                # nearest_pos = self.mesh_world.nearest.on_surface(np.asarray(nominal_pos)[np.newaxis, :])
                # embed()
                current_tip_poses[arm] = util.pose_from_vectors(x_vec, y_vec, z_vec, nearest_pos)
                # if arm == 'right':
                #     embed()
        if primitive_name == 'pivot':

            start = time.time()
            des_normals = {}
            self.state_lock.acquire()
            des_normals['left'] = util.C3(3*self._dtheta_left).dot(self._left_normal_vec.T)
            des_normals['right'] = util.C3(self._dtheta_right).dot(self._right_normal_vec.T)
            # des_normals['right'] = self._right_normal_vec.T
            dn1 = self._dn1
            # palm_z_normal = self._forward_normal
            palm_z_normal = np.array([1, 0, 0])
            print('Delta Theta Left: ' + str(self._dtheta_left) + ' Delta Theta Right: ' + str(self._dtheta_right))
            dtheta = {}
            dtheta['right'] = self._dtheta_right
            dtheta['left'] = self._dtheta_left
            self.state_lock.release()

            for arm in ['right', 'left']:
                current_pos = util.pose_stamped2list(current_tip_poses[arm])[:3]
                nom = self.cfg.PALM_RIGHT if arm == 'right' else self.cfg.PALM_LEFT
                nominal_pos = util.pose_stamped2list(
                    util.convert_reference_frame(
                        util.list2pose_stamped(nom),
                        util.unit_pose(),
                        object_pose_current))[:3]
                self.transform_mesh_world()
                # nearest_pos = self.mesh_world.nearest.on_surface(np.asarray(current_pos)[np.newaxis, :])[0][0].tolist()
                # nearest_pos = self.mesh_world.nearest.on_surface(np.asarray(nominal_pos)[np.newaxis, :])[0][0].tolist()

                #could also keep track of where the midpoint between the two vertices of interest are... try this next...

                # current_tip_poses[arm] = util.list2pose_stamped(
                #     nearest_pos + util.pose_stamped2list(current_tip_poses[arm])[3:])

                # y_vec = des_normals[arm]
                # z_vec = palm_z_normal
                # x_vec = np.cross(y_vec, z_vec)
                # current_tip_poses[arm] = util.pose_from_vectors(
                #     x_vec, y_vec, z_vec, nearest_pos
                # )

                # y_vec = des_normals[arm]
                # z_vec = palm_z_normal
                # x_vec = np.cross(y_vec, z_vec)
                # current_tip_poses[arm] = util.pose_from_vectors(
                #     x_vec, y_vec, z_vec, nominal_pos
                # )
                if arm == 'right':
                #     d_pos = self.rotate_palm_about_point(dtheta)
                #     current_pos[0] += d_pos['right'][0]
                #     current_pos[1] += d_pos['right'][1]
                #     current_pos[2] += d_pos['right'][2]
                    displacement_vec = (dn1/300.0) * np.asarray(des_normals[arm])
                    # current_pos = (np.asarray(nominal_pos) - displacement_vec).tolist()
                    current_pos = (np.asarray(current_pos) - displacement_vec).tolist()
                    # print('dn1, disp vec: ' + str(dn1), displacement_vec)
                y_vec = des_normals[arm]
                z_vec = palm_z_normal
                x_vec = np.cross(y_vec, z_vec)
                current_tip_poses[arm] = util.pose_from_vectors(
                    x_vec, y_vec, z_vec, current_pos
                )

            # print('angle reset time: ' + str(time.time() - start))


        # if primitive_name == 'grasp':
        #     normals = self.get_palm_y_normals()
        #     self.state_lock.acquire()
        #     ds = self._delta_spring
        #     self.state_lock.release()
        #     for arm in ['right', 'left']:
        #         current_pos = util.pose_stamped2list(current_tip_poses[arm])[:3]
        #         normal = np.asarray(util.pose_stamped2list(normals[arm])[:3]) - np.asarray(current_pos)
        #         # self.state_lock.acquire()
        #         # self.transform_mesh_world()
        #         # nearest_pos = self.mesh_world.nearest.on_surface(np.asarray(current_pos)[np.newaxis, :])[0][0].tolist()
        #         # self.state_lock.release()
        #         # if arm == 'right':
        #         current_pos[0] += ds*normal[0]
        #         current_pos[1] += ds*normal[1]
        #         current_pos[2] += ds*normal[2]
        #         # current_tip_poses[arm] = util.list2pose_stamped(
        #         #     nearest_pos + util.pose_stamped2list(current_tip_poses[arm])[3:])
        #         current_tip_poses[arm] = util.list2pose_stamped(
        #             current_pos + util.pose_stamped2list(current_tip_poses[arm])[3:])

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

        primitive_args = {}
        primitive_args['object_pose1_world'] = object_pose_current
        primitive_args['object_pose2_world'] = self.object_pose_final
        primitive_args['palm_pose_l_object'] = l_tip_pose_object_frame
        primitive_args['palm_pose_r_object'] = r_tip_pose_object_frame
        primitive_args['object'] = None
        # if primitive_name == 'pivot':
        #     primitive_args['N'] = 50
        # else:
        #     primitive_args['N'] = int(
        #         len(self.initial_plan[plan_number]['palm_poses_world'])*frac_done)
        primitive_args['N'] = int(
            len(self.initial_plan[plan_number]['palm_poses_world'])*frac_done)
        primitive_args['init'] = False
        primitive_args['table_face'] = self.table_face

        start = time.time()
        new_plan = self.get_primitive_plan(
            primitive_name,
            primitive_args,
            self.active_arm)
        # print('new plan time: ' + str(time.time() - start))

        # self.robot.set_sim_sleep(30)
        # if simulate and frac_done < 0.5:
        #     import simulation
        #     for i in range(10):
        #         simulation.visualize_object(
        #             object_pose_current,
        #             filepath="package://config/descriptions/meshes/objects/realsense_box_experiments.stl",
        #             name="/object_initial",
        #             color=(1., 0., 0., 1.),
        #             frame_id="/yumi_body",
        #             scale=(1., 1., 1.))
        #         simulation.visualize_object(
        #             self.object_pose_final,
        #             filepath="package://config/descriptions/meshes/objects/realsense_box_experiments.stl",
        #             name="/object_final",
        #             color=(0., 0., 1., 1.),
        #             frame_id="/yumi_body",
        #             scale=(1., 1., 1.))
        #         rospy.sleep(.1)
        #     simulation.simulate(new_plan, 'realsense_box_experiments.stl')
        # self.robot.set_sim_sleep(0.3)

        if primitive_name == 'grasp':
            next_step = 2 if (plan_number == 0 or plan_number == 1) else 14
        elif primitive_name == 'pivot':
            next_step = 2
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
                        pos_tol=0.01, ori_tol=0.02,
                        rs=False, z=False):
        """
        Check if manipulated object reached goal or not. Returns true
        if both position and orientation goals have been reached
        within specified tolerance

        Args:
            pos (list np.ndarray): goal position
            ori (list or np.ndarray): goal orientation. It can be:
                **quaternion** ([qx, qy, qz, qw], shape: :math:`[4]`)
                **rotation matrix** (shape: :math:`[3, 3]`)
                **euler angles** ([roll, pitch, yaw], shape: :math:`[3]`)
            object_id (int): PyBullet object id of the object being
                manipulated
            pos_tol (float): tolerance of position error
            ori_tol (float): tolerance of orientation error


        Returns:
            3-element tupe containing:
            - bool: If goal pose is reached or not
            - float: The position error
            - float: The orientation error
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

        if z:
            pos_diff = new_ee_pos.flatten() - goal_pos
        else:
            pos_diff = new_ee_pos.flatten()[:-1] - goal_pos[:-1]
        # pos_error = np.max(np.abs(pos_diff))
        pos_error = np.linalg.norm(pos_diff)

        quat_diff = common.quat_multiply(common.quat_inverse(goal_ori),
                                         new_ee_quat)
        rot_similarity = np.abs(quat_diff[3])
        dot_prod = np.dot(goal_ori, new_ee_quat)
        angle_diff = np.arccos(2*dot_prod**2 - 1)

        # if pos_error < pos_tol and \
        #         rot_similarity > 1 - ori_tol:
        #     return True, pos_error, 1-rot_similarity
        # else:
        #     return False, pos_error, 1-rot_similarity
        reached = False
        if rs:
            if pos_error < pos_tol and rot_similarity > 1 - ori_tol:
                reached = True
                angle_diff = 1 - rot_similarity
        else:
            if pos_error < pos_tol and angle_diff < ori_tol:
                reached = True
        return reached, pos_error, angle_diff

    def add_remove_scene_object(self, action='add'):
        """
        Helper function to add or remove an object from the MoveIt
        planning scene

        Args:
            action (str, optional): Whether to 'add' or 'remove'
                the object. Defaults to 'add'.
        """
        if action != 'add' and action != 'remove':
            raise ValueError('Action not recognied, must be either'
                             'add or remove')

        if action == 'add':
            # print("GETTING POSE")
            object_pos_world = list(p.getBasePositionAndOrientation(
                self.object_id,
                self.pb_client)[0])
            object_ori_world = list(p.getBasePositionAndOrientation(
                self.object_id,
                self.pb_client)[1])

            pose = util.list2pose_stamped(
                object_pos_world + object_ori_world, "yumi_body")
            pose_stamped = PoseStamped()

            # print("MAKING POSE STAMPED")
            pose_stamped.header.frame_id = pose.header.frame_id
            pose_stamped.pose.position.x = pose.pose.position.x
            pose_stamped.pose.position.y = pose.pose.position.y
            pose_stamped.pose.position.z = pose.pose.position.z
            pose_stamped.pose.orientation.x = pose.pose.orientation.x
            pose_stamped.pose.orientation.y = pose.pose.orientation.y
            pose_stamped.pose.orientation.z = pose.pose.orientation.z
            pose_stamped.pose.orientation.w = pose.pose.orientation.w

            # print("MOVEIT SCENE ADD MESH")
            # from IPython import embed
            # embed()
            self.robot.moveit_scene.add_mesh(
                name='object',
                pose=pose_stamped,
                filename=self.object_mesh_file,
                size=(0.975, 0.975, 0.975)
            )
        elif action == 'remove':
            self.robot.moveit_scene.remove_world_object(
                name='object'
            )

    def full_mp_check(self, initial_plan, primitive_name,
                      start_joints=None):
        """
        Run the full plan through motion planning to check feasibility
        right away, don't execute if any of the steps fail motion planning

        Args:
            initial_plan (list): List of plan dictionaries returned
                by primitive planner
            primitive_name (str): Type of primitive being executed,
                determines whether both arms or only single arm should
                be checked
            start_joints (dict): Dictionary of nominal starting joint configuration,
                with 'right' and 'left' keys. If None, will use current
                robot joint configuration from simulator

        Returns:
            bool: True if full plan is feasible/valid, else False
        """
        right_valid = []
        left_valid = []

        for subplan_number, subplan_dict in enumerate(initial_plan):
            subplan_tip_poses = subplan_dict['palm_poses_world']

            # setup motion planning request with all the cartesian waypoints
            tip_right = []
            tip_left = []

            # bump y a bit in the palm frame for pre pose, for collision avoidance
            if subplan_number == 0:
                pre_pose_right_init = util.unit_pose()
                pre_pose_left_init = util.unit_pose()

                pre_pose_right_init.pose.position.y += 0.05
                pre_pose_left_init.pose.position.y += 0.05

                pre_pose_right = util.transform_pose(
                    pre_pose_right_init, subplan_tip_poses[0][1])

                pre_pose_left = util.transform_pose(
                    pre_pose_left_init, subplan_tip_poses[0][0])

                # tip_right.append(
                #     self.robot.compute_fk(self.cfg.RIGHT_INIT, arm='right').pose)
                # tip_left.append(
                #     self.robot.compute_fk(self.cfg.LEFT_INIT, arm='left').pose)

                tip_right.append(pre_pose_right.pose)
                tip_left.append(pre_pose_left.pose)

            for i in range(len(subplan_tip_poses)):
                tip_right.append(subplan_tip_poses[i][1].pose)
                tip_left.append(subplan_tip_poses[i][0].pose)

            if start_joints is None:
                l_start = self.robot.get_jpos(arm='left')
                r_start = self.robot.get_jpos(arm='right')
            else:
                l_start = start_joints['left']
                r_start = start_joints['right']

            # l_start = self.robot.compute_ik(
            #     pos=util.pose_stamped2list(subplan_tip_poses[0][0])[:3],
            #     ori=util.pose_stamped2list(subplan_tip_poses[0][0])[3:],
            #     seed=self.robot.get_jpos(arm='left'),
            #     arm='left'
            # )
            # r_start = self.robot.compute_ik(
            #     pos=util.pose_stamped2list(subplan_tip_poses[0][1])[:3],
            #     ori=util.pose_stamped2list(subplan_tip_poses[0][1])[3:],
            #     seed=self.robot.get_jpos(arm='right'),
            #     arm='right'
            # )

            # motion planning for both arms
            # if self.object_mesh_file is not None:
            #     self.add_remove_scene_object(action='add')

            try:
                self.robot.mp_right.plan_waypoints(
                    tip_right,
                    force_start=l_start+r_start,
                    avoid_collisions=True
                )
                right_valid.append(1)
            except ValueError as e:
                break
            try:
                self.robot.mp_left.plan_waypoints(
                    tip_left,
                    force_start=l_start+r_start,
                    avoid_collisions=True
                )
                left_valid.append(1)
            except ValueError as e:
                break
        valid = False
        if primitive_name == 'grasp' or primitive_name == 'pivot':
            if sum(right_valid) == len(initial_plan) and \
                    sum(left_valid) == len(initial_plan):
                valid = True
        else:
            # if self.active_arm == 'right':
            if sum(right_valid) == len(initial_plan):
                valid = True
            # else:
            #     if sum(left_valid) == len(self.initial_plan):
            #         valid = True
        return valid

    def pushing_normal_alignment(self, show=True):
        # wrist_poses = {}
        # out_obj, out_palm = 0, 0
        # for arm in ['right', 'left']:
        #     wrist_pos_world = self.robot.get_ee_pose(arm=arm)[0].tolist()
        #     wrist_ori_world = self.robot.get_ee_pose(arm=arm)[1].tolist()

        #     wrist_poses[arm] = util.list2pose_stamped(wrist_pos_world + wrist_ori_world)

        # tip_poses = self.robot.wrist_to_tip(wrist_poses)
        # object_pos = list(p.getBasePositionAndOrientation(
        #     self.object_id, self.pb_client)[0])
        # object_ori = list(p.getBasePositionAndOrientation(
        #     self.object_id, self.pb_client)[1])
        # object_pose = util.list2pose_stamped(object_pos + object_ori)

        # pos1 = util.pose_stamped2list(util.convert_reference_frame(util.list2pose_stamped(self.cfg.PALM_RIGHT), util.unit_pose(), object_pose))[:3]
        # # pos1 = util.pose_stamped2list(tip_poses['right'])[:3]
        # self.transform_mesh_world()
        # pos2 = self.mesh_world.face_normals[0]
        # pos3 = pos1 + pos2

        # if show:
        #     out_obj = p.addUserDebugLine(pos1, pos3, [0, 1, 0])

        # palm_normal = self.get_palm_y_normals(None, current=True)['right']

        # pos1 = util.pose_stamped2list(tip_poses['right'])[:3]
        # pos2 = util.pose_stamped2list(palm_normal)[:3]

        # if show:
        #     out_palm = p.addUserDebugLine(pos1, pos2, [0, 0, 1])

        # palm_normal_rel = np.asarray(pos2) - np.asarray(pos1)

        # err = -(1 - np.dot(palm_normal_rel, obj_normal_rel))
        # theta = np.arccos(np.dot(palm_normal_rel, obj_normal_rel))
        # print('theta: ' + str(theta))
        #dq = self.kp*err + self.kd*(last_err - err)
        # dq = 10*err + self.kd*(last_err - err)

        self.state_lock.acquire()
        self.transform_mesh_world()
        obj_normal_rel = self.mesh_world.face_normals[0]
        self.state_lock.release()

        y_vec = obj_normal_rel
        z_vec = np.array([0, 0, -1])
        x_vec = np.cross(y_vec, z_vec)

        vectors = [x_vec, y_vec, z_vec]

        # tip_ori = util.pose_from_vectors(x_vec, y_vec, z_vec, [0, 0, 0])
        # ori_list = util.pose_stamped2list(tip_ori)[3:]

        # return err, theta, vectors, out_obj, out_palm
        return None, None, vectors, None, None

    def leaky_integrator(self, prev, L, primitive='grasp'):
        if primitive == 'grasp':
            step = -0.00025
            max_deviation = 0.025
        elif primitive == 'pull':
            step = -0.0005
            # step = 0.0
            max_deviation = 0.025
        alpha = 0.99
        if L == 1:
            new = prev + step
        else:
            new = alpha * prev + (1 - alpha) * L
        return max(-max_deviation, new)

    def rotate_palm_about_point(self, dtheta):
        object_pos = list(p.getBasePositionAndOrientation(
            self.object_id, self.pb_client)[0])
        object_ori = list(p.getBasePositionAndOrientation(
            self.object_id, self.pb_client)[1])
        object_pose = util.list2pose_stamped(object_pos + object_ori)

        right_center = util.pose_stamped2list(
            util.convert_reference_frame(
                util.list2pose_stamped(self.cfg.PALM_RIGHT),
                util.unit_pose(),
                object_pose))[:3]

        left_center = util.pose_stamped2list(
            util.convert_reference_frame(
                util.list2pose_stamped(self.cfg.PALM_LEFT),
                util.unit_pose(),
                object_pose))[:3]

        current_tip_poses = self.get_current_tip_poses()
        current_right_pos = util.pose_stamped2list(current_tip_poses['right'])[:3]
        current_left_pos = util.pose_stamped2list(current_tip_poses['left'])[:3]

        ry = current_right_pos[1] - right_center[1]
        rz = current_right_pos[2] - right_center[2]
        r = np.sqrt(ry**2 + rz**2)

        # desired angle rotation
        self.state_lock.acquire()
        current_right_angle = self._right_palm_angle
        self.state_lock.release()
        des_right_angle = current_right_angle + dtheta['right']

        dy = r * (np.cos(des_right_angle + np.pi) - np.cos(current_right_angle + np.pi))
        dz = r * (np.sin(des_right_angle + np.pi) - np.sin(current_right_angle + np.pi))
        # dy = r * np.cos(dtheta['right'])
        # dz = r * np.sin(dtheta['right'])
        dx = 0

        delta_pos = {}
        delta_pos['right'] = [dx, dy, dz]
        delta_pos['left'] = [0, 0, 0]

        # if dtheta['right'] != 0.0:
        #     embed()
        return delta_pos

    def palm_object_state_estimator(self):
        self._dtheta_right = 0
        self._dtheta_left = 0
        self._left_normal_vec = np.array([0, 0, 1])
        self._right_normal_vec = np.array([0, 0, 1])
        self._forward_normal = np.array([1, 0, 0])
        self._tactile_in_contact = False
        self._tactile_state = None
        self._tactile_control = None
        self._delta_spring = 0.0
        self._dn1 = 0.0
        delta_spring = 0.0
        t_thresh = 0.5
        n_thresh = 4.0
        state = None
        control = None
        while True:

            start = time.time()
            tip_poses = self.get_current_tip_poses()
            # 3 and 2 are top face in example config
            # face 3 for right palm, where start would be + np.pi/2
            # face 2 for left palm, where start would be -np.pi/2

            # 6 is the face on contact with left palm at start config
            # 8 is face in contact with right palm at start config

            left_current_pos = np.asarray(util.pose_stamped2list(tip_poses['left'])[:3])
            right_current_pos = np.asarray(util.pose_stamped2list(tip_poses['right'])[:3])

            palm_normal_poses = self.get_palm_y_normals()

            left_normal_vec = np.asarray(util.pose_stamped2list(palm_normal_poses['left'])[:3]) - left_current_pos
            right_normal_vec = np.asarray(util.pose_stamped2list(palm_normal_poses['right'])[:3]) - right_current_pos

            self.state_lock.acquire()
            self.transform_mesh_world()
            object_normal_left = self.mesh_world.face_normals[6]
            object_normal_right = self.mesh_world.face_normals[8]
            forward_normal = self.mesh_world.face_normals[-1]
            vertices = {}
            for i, vertex in enumerate(self.mesh_world.vertices):
                vertices[i] = vertex
            self.state_lock.release()

            left_angle = np.arccos(np.dot(object_normal_left/np.linalg.norm(object_normal_left), left_normal_vec/np.linalg.norm(left_normal_vec)))
            right_angle = np.arccos(np.dot(object_normal_right/np.linalg.norm(object_normal_right), right_normal_vec/np.linalg.norm(right_normal_vec)))

            cross_left = np.cross(object_normal_left, left_normal_vec)
            cross_right = np.cross(object_normal_right, right_normal_vec)

            # and then get front face for positive x axis to compare sign of, for cross product step
            if np.dot(forward_normal, cross_left) < 0:
                left_angle = -left_angle
            if np.dot(forward_normal, cross_right) < 0:
                right_angle = -right_angle

            obj_angle = np.arccos(np.dot(object_normal_left, [0, 1, 0]))
            if np.dot(forward_normal, np.cross([0, 1, 0], object_normal_left)) < 0:
                obj_angle = -obj_angle

            flags = {'normal': -7, 'contact_distance': -6, 'normal_f': -5,
                     'lateral_1_f': -4, 'lateral_1': -3,
                     'lateral_2_f': -2, 'lateral_2': -1}

            table_object, left_object, right_object = [], [], []
            # get contact forces between table and object
            table_object_n = 0
            for i, pt in enumerate(p.getContactPoints(self.robot.yumi_pb.arm.robot_id, self.object_id, self.cfg.TABLE_ID, -1)):
                table_object_n += np.abs(pt[-5])
                table_object.append({})
                for key, val in flags.items():
                    table_object[i][key] = pt[val]


            # get contact forces between table and object
            right_object_n_list = []
            right_object_t1 = []
            right_object_t2 = []
            for i, pt in enumerate(p.getContactPoints(self.robot.yumi_pb.arm.robot_id, self.object_id, self.cfg.RIGHT_GEL_ID, -1)):
                right_object_n_list.append(np.abs(pt[-5]))
                right_object_t1.append(np.abs(pt[-4]))
                right_object_t2.append(np.abs(pt[-2]))
                right_object.append({})
                for key, val in flags.items():
                    right_object[i][key] = pt[val]
                # print('right c')
            right_object_n = sum(right_object_n_list)

            # get contact forces between table and object
            left_object_n = 0
            for i, pt in enumerate(p.getContactPoints(self.robot.yumi_pb.arm.robot_id, self.object_id, self.cfg.LEFT_GEL_ID, -1)):
                left_object_n += np.abs(pt[-5])
                left_object.append({})
                for key, val in flags.items():
                    left_object[i][key] = pt[val]
                # print('left c')

            obj_pos_world = list(p.getBasePositionAndOrientation(
                self.object_id, self.pb_client)[0])
            obj_ori_world = list(p.getBasePositionAndOrientation(
                self.object_id, self.pb_client)[1])

            self.object_poses_list.append(obj_pos_world + obj_ori_world)
            self.table_object_contact_list.append(table_object)
            self.right_object_contact_list.append(right_object)
            self.left_object_contact_list.append(left_object)
            self.vertex_positions_list.append(vertices)
            self.tip_poses_list.append(tip_poses)

            new_dict = {}
            new_dict['dtheta1'] = 0
            new_dict['dtheta2'] = 0
            new_dict['dn1'] = 0
            if self.primitive_name == 'pivot':
                state = [0, 0, obj_angle]
                control = [right_object_n, right_angle, left_angle]
            elif self.primitive_name == 'pull':
                state = [0, 0, 0]
                control = [right_object_n, 0, 0]
            if self._tactile_in_contact:
                try:
                    self.tactile_control.solve_equilibrium(
                        control_input=control,
                        state_vec=state,
                        is_show=False)

                    new_dict = self.tactile_control.solve_controller(
                        state_vec=state,
                        is_show=False)
                    if self.primitive_name in ['pull', 'grasp']:
                        # L = new_dict['f1'][1] > self.tactile_control.model_equilibrium.state_equilibrium_dict['f1'][1]
                        # L = max(right_object_t1) > t_thresh or max(right_object_t2) > t_thresh or right_object_n < n_thresh
                        # L = right_object_n < n_thresh
                        # L = False
                        kvs = {}
                        # kvs['L'] = L
                        kvs['t1'] = max(right_object_t1)
                        kvs['t2'] = max(right_object_t2)
                        kvs['n_tot'] = right_object_n
                        kvs['n_max'] = max(right_object_n_list)
                        kvs['alpha1'] = max(np.asarray(right_object_n) - np.asarray(right_object_t1)/0.5)
                        kvs['alpha2'] = max(np.asarray(right_object_n) - np.asarray(right_object_t2)/0.5)
                        L = kvs['alpha1'] < 12.0 or kvs['alpha2'] < 12.0 #or kvs['t1'] > t_thresh or kvs['t2'] > t_thresh
                        string = " "

                        for k, v in kvs.items():
                            string += "%s: %.3f, " % (k,v)
                        print("L: " + str(L) + string)
                        # print('L, t1, t2, n: ' +
                        #       str(L) + ' ' +
                        #       str(max(right_object_t1)) + ' ' +
                        #       str(max(right_object_t2)) + ' ' +
                        #       str(right_object_n))
                        delta_spring = self.leaky_integrator(delta_spring, L, primitive=self.primitive_name)
                        new_dict['dtheta1'] = 0
                        new_dict['dtheta2'] = 0
                        new_dict['dn1'] = 0
                except:
                    new_dict['dtheta1'] = 0
                    new_dict['dtheta2'] = 0
                    new_dict['dn1'] = 0
                    if self.primitive_name in ['pull', 'grasp']:
                        delta_spring = self.leaky_integrator(delta_spring, 0, primitive=self.primitive_name)
            # print('right angle: ' + str(right_angle) +
            #         ' left angle: ' + str(left_angle) +
            #         ' object angle: ' + str(obj_angle)
            # )
            if self.primitive_name in ['pull', 'grasp', 'pivot']:
                self.state_lock.acquire()
                self._delta_spring = delta_spring
                self._right_palm_angle = right_angle
                self._left_palm_angle = left_angle
                self._object_angle = obj_angle
                self._dtheta_right = new_dict['dtheta1']
                self._dtheta_left = new_dict['dtheta2']
                self._dn1 = new_dict['dn1']
                self._left_normal_vec = left_normal_vec
                self._right_normal_vec = right_normal_vec
                self._forward_normal = forward_normal
                self._tactile_state = state
                self._tactile_control = control
                self.state_lock.release()
            # print('tactile thread time: ' + str(time.time() - start))

            time.sleep(0.1)

    def execute_single_arm(self, primitive_name, subplan_dict,
                           subplan_goal, subplan_number):
        """
        Function to execute a single arm primitive action

        Args:
            primitive_name (str): Which primitive to use
            subplan_dict (dict): The nominal initial plan for
                this subplan of this primitive
            subplan_goal (PoseStamped): The intermediate goal
                pose of the object for this subplan
            subplan_number (int): The index of this subplan w.r.t.
                the overall plan

        Returns:
            3-element tuple containing:
            - bool: Whether the goal was reached or not. True if goal
              reached within desired tolerance, else False
            - float: position error at the end
            - float: orientation error at the end
        """
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

        # TODO why am I doing this?
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

        # bump z a bit for pulling for collision avoidance on the approach
        # if primitive_name == 'pull':
        #     pre_pose_right = copy.deepcopy(subplan_tip_poses[0][1].pose)
        #     pre_pose_left = copy.deepcopy(subplan_tip_poses[0][0].pose)

        #     pre_pose_right.position.z += 0.1
        #     pre_pose_left.position.z += 0.1

        #     tip_right.append(pre_pose_right)
        #     tip_left.append(pre_pose_left)
        pre_pose_right_init = util.unit_pose()
        pre_pose_left_init = util.unit_pose()

        pre_pose_right_init.pose.position.y += 0.05
        pre_pose_left_init.pose.position.y += 0.05

        pre_pose_right = util.transform_pose(
            pre_pose_right_init, subplan_tip_poses[0][1])

        pre_pose_left = util.transform_pose(
            pre_pose_left_init, subplan_tip_poses[0][0])

        # tip_right.append(
        #     self.robot.compute_fk(self.cfg.RIGHT_INIT, arm='right').pose)
        # tip_left.append(
        #     self.robot.compute_fk(self.cfg.LEFT_INIT, arm='left').pose)

        tip_right.append(pre_pose_right.pose)
        tip_left.append(pre_pose_left.pose)

        for i in range(len(subplan_tip_poses)):
            tip_right.append(subplan_tip_poses[i][1].pose)
            tip_left.append(subplan_tip_poses[i][0].pose)

        l_current = self.robot.get_jpos(arm='left')
        r_current = self.robot.get_jpos(arm='right')
        # l_current = self.cfg.LEFT_INIT
        # r_current = self.cfg.RIGHT_INIT

        # self.add_remove_scene_object(action='add')

        # call motion planning to get trajectory without large joint jumps
        if self.active_arm == 'right':
            traj = self.robot.mp_right.plan_waypoints(
                tip_right,
                force_start=l_current+r_current,
                avoid_collisions=True
            )
        else:
            traj = self.robot.mp_left.plan_waypoints(
                tip_left,
                force_start=l_current+r_current,
                avoid_collisions=True
            )
        # self.add_remove_scene_object(action='remove')

        # make numpy array of each arm joint trajectory for each comp
        joints_np = np.zeros((len(traj.points), 7))

        # make numpy array of each arm pose trajectory, based on fk
        fk_np = np.zeros((len(traj.points), 7))

        # make a Cartesian version of the path
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
        still_in_contact = False
        slipping = False
        slipped = 0
        last_err = 0
        open_loop_switch = False
        set_time = False
        perturbed = False
        perturbing = False
        perturbed_n = 0
        # pos_errs, ori_errs = [], []
        errs = []
        while not reached_goal:
            # get closest point in nominal plan to where
            # the robot currently is in the world
            pose_ref = util.pose_stamped2list(
                self.robot.compute_fk(
                    joints=self.robot.get_jpos(arm=self.active_arm),
                    arm=self.active_arm))
            diffs = util.pose_difference_np(
                pose=fk_np,
                pose_ref=pose_ref)[0]

            # use this as the seed for greedy replanning based on IK
            seed_ind = min(np.argmin(diffs), joints_np.shape[0]-2)

            seed = {}
            seed[self.active_arm] = joints_np[seed_ind, :]
            seed[self.inactive_arm] = \
                self.robot.get_jpos(arm=self.inactive_arm)

            if made_contact and not set_time and primitive_name == 'push' and self.replan:
                self.robot.set_sim_sleep(0.1)
                set_time = True

            perturb_valid = made_contact and \
                not perturbed and not perturbing and \
                np.random.random() > 0.7 and \
                seed_ind > joints_np.shape[0]/2.5 and self.perturb #and \
                # primitive_name == 'pull' and self.perturb
            # perturb_valid = False
            if perturb_valid:
                perturbing = True
                f1 = np.random.random()*0.5
                f2 = np.random.random()*2 + 0.5
                f1 = -f1 if np.random.random() > 0.5 else f1
                f2 = -f2 if np.random.random() > 0.5 else f2
                tau = np.random.random()*0.5 - 0.25
                # f = 7.75/1.0
                # tau = -0.25/1.0
                f_vec = [f1, 0, 0]
                tau_vec = [0, 0, tau]
                print('\n\n\nApplying perturbation! Force: %f, Torque: %f\n\n\n' % (f1, tau))

            if perturbing:
                obj_pos = p.getBasePositionAndOrientation(self.object_id)[0]
                p.applyExternalForce(
                    self.object_id,
                    -1,
                    f_vec,
                    obj_pos,
                    p.WORLD_FRAME)
                # p.applyExternalTorque(
                #     self.object_id,
                #     -1,
                #     tau_vec,
                #     p.WORLD_FRAME
                # )
                perturbed_n += 1
                self.perturbations_list.append((f_vec, len(self.object_poses_list)))
                if perturbed_n > 15:
                    perturbed = True
                    perturbing = False
                    print('\n\n\nPerturbation complete!\n\n\n')

            if self.replan and made_contact:
                ik_iter = 0
                ik_sol_found = False
                while not ik_sol_found:
                    joints_execute, new_plan = self.greedy_replan(
                        primitive_name=primitive_name,
                        object_id=self.object_id,
                        seed=seed,
                        plan_number=subplan_number,
                        frac_done=pos_err/pos_err_total)
                    joints_execute = joints_execute[self.active_arm]
                    if joints_execute is not None:
                        ik_sol_found = True
                    ik_iter += 1

                    if ik_iter > self.max_ik_iter:
                        ind1 = np.argmin(np.asarray(errs)[:, 0], axis=0)
                        ind2 = np.argmin(np.asarray(errs)[:, 1], axis=0)
                        print('MIN ERRS: ', errs[ind1], errs[ind2])                        
                        raise ValueError('IK solution not found!')
            else:
                joints_execute = joints_np[seed_ind+1, :].tolist()

            self.robot.update_joints(joints_execute, arm=self.active_arm)

            if primitive_name == 'push' and self.replan:
                reached_goal, pos_err, ori_err = self.reach_pose_goal(
                    subplan_goal[:3],
                    subplan_goal[3:],
                    self.object_id,
                    pos_tol=0.01, ori_tol=0.1)
                # print(pos_err, ori_err)
                # pos_errs.append(pos_err)
                # ori_errs.append(ori_err)
                # errs.append([pos_err, ori_err])
            else:
                reached_goal, pos_err, ori_err = self.reach_pose_goal(
                    subplan_goal[:3],
                    subplan_goal[3:],
                    self.object_id,
                    pos_tol=self.goal_pos_tol, ori_tol=self.goal_ori_tol)
                # print(ori_err)
            errs.append([pos_err, ori_err])

            if (seed_ind == joints_np.shape[0]-2) and not self.replan:
                reached_goal = True

            timed_out = time.time() - start_time > self.subgoal_timeout
            if timed_out and primitive_name == 'pull':
                print("TIMED OUT!")
                break
            time.sleep(0.05)
            if not made_contact:
                made_contact = self.robot.is_in_contact(self.object_id)[self.active_arm]
                self._tactile_in_contact = True
            if made_contact:
                still_in_contact = self.robot.is_in_contact(self.object_id)[self.active_arm]
                slipping = still_in_contact
                if not still_in_contact:
                    slipped += 1
            if slipped > 15:
                print("LOST CONTACT!")
                break
        ind1 = np.argmin(np.asarray(errs)[:, 0], axis=0)
        ind2 = np.argmin(np.asarray(errs)[:, 1], axis=0)
        print('MIN ERRS: ', errs[ind1], errs[ind2])
        return reached_goal, pos_err, ori_err

    def execute_two_arm(self, primitive_name, subplan_dict,
                        subplan_goal, subplan_number):
        """
        Function to execute a dual arm primitive action

        Args:
            primitive_name (str): Which primitive to use
            subplan_dict (dict): The nominal initial plan for
                this subplan of this primitive
            subplan_goal (PoseStamped): The intermediate goal
                pose of the object for this subplan
            subplan_number (int): The index of this subplan w.r.t.
                the overall plan

        Returns:
            3-element tuple containing:
            - bool: Whether the goal was reached or not. True if goal
              reached within desired tolerance, else False
            - float: position error at the end
            - float: orientation error at the end
        """
        subplan_tip_poses = subplan_dict['palm_poses_world']

        # setup motion planning request with all the cartesian waypoints
        tip_right = []
        tip_left = []

        # bump y a bit in the palm frame for pre pose, for collision avoidance
        if subplan_number == 0:
            pre_pose_right_init = util.unit_pose()
            pre_pose_left_init = util.unit_pose()

            pre_pose_right_init.pose.position.y += 0.05
            pre_pose_left_init.pose.position.y += 0.05

            pre_pose_right = util.transform_pose(
                pre_pose_right_init, subplan_tip_poses[0][1])

            pre_pose_left = util.transform_pose(
                pre_pose_left_init, subplan_tip_poses[0][0])

            # tip_right.append(
            #     self.robot.compute_fk(self.cfg.RIGHT_INIT, arm='right').pose)
            # tip_left.append(
            #     self.robot.compute_fk(self.cfg.LEFT_INIT, arm='left').pose)

            tip_right.append(pre_pose_right.pose)
            tip_left.append(pre_pose_left.pose)

        for i in range(len(subplan_tip_poses)):
            tip_right.append(subplan_tip_poses[i][1].pose)
            tip_left.append(subplan_tip_poses[i][0].pose)

        # bump y a bit in the palm frame for pre pose, for not throwing object
        if subplan_number == 2:
            # post_pose_right_init = util.unit_pose()
            # post_pose_left_init = util.unit_pose()

            # post_pose_right_init.pose.position.y += 0.05
            # post_pose_left_init.pose.position.y += 0.05

            # post_pose_right = util.transform_pose(
            #     post_pose_right_init, subplan_tip_poses[-1][1])

            # post_pose_left = util.transform_pose(
            #     post_pose_left_init, subplan_tip_poses[-1][0])

            # tip_right.append(post_pose_right.pose)
            # tip_left.append(post_pose_left.pose)
            tip_right[-1].position.y -= 0.05
            tip_left[-1].position.y += 0.05

        l_start = self.robot.get_jpos(arm='left')
        r_start = self.robot.get_jpos(arm='right')

        # l_start = self.cfg.LEFT_INIT
        # r_start = self.cfg.RIGHT_INIT

        # motion planning for both arms
        # if self.object_mesh_file is not None:
        #     self.add_remove_scene_object(action='add')

        try:
            traj_right = self.robot.mp_right.plan_waypoints(
                tip_right,
                force_start=l_start+r_start,
                avoid_collisions=True
            )
        except ValueError as e:
            # print(e)
            # print("right arm")
            # if subplan_number == 2:
            #     embed()
            raise ValueError(e)

        try:
            traj_left = self.robot.mp_left.plan_waypoints(
                tip_left,
                force_start=l_start+r_start,
                avoid_collisions=True
            )
        except ValueError as e:
            # print(e)
            # print("left arm")
            # if subplan_number == 2:
            #     embed()
            raise ValueError(e)

        # if self.object_mesh_file is not None:
        #     self.add_remove_scene_object(action='remove')

        # after motion planning, unify the dual arm trajectories
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
            pos_tol=self.goal_pos_tol, ori_tol=self.goal_ori_tol, 
            rs=True,
            z=True)
        # print(reached_goal, pos_err_total, ori_err_total)
        # embed()

        pos_err = pos_err_total
        ori_err = ori_err_total

        timed_out = False
        start_time = time.time()

        ended = False
        set_time = False

        prev_joints = self.robot.get_jpos()
        prev_seed_ind_r, prev_seed_ind_l = 0, 0

        perturbed = False
        perturbing = False
        perturbed_n = 0

        errs = []        
        while not reached_goal and not ended:
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

            # seed_ind_r = min(np.argmin(diffs_r[0]), aligned_right.shape[0]-2)
            # seed_ind_l = min(np.argmin(diffs_l[0]), aligned_left.shape[0]-2)
            seed_ind_r = min(np.argmin(diffs_r[0][:prev_seed_ind_r+10]), aligned_right.shape[0]-2)
            seed_ind_l = min(np.argmin(diffs_l[0][:prev_seed_ind_l+10]), aligned_left.shape[0]-2)            

            # make sure arm is not just going back in forth, if it's been stuck a while
            # push it forward
            # repeat_count_r[seed_ind_r] += 1
            # repeat_count_l[seed_ind_l] += 1
            # if repeat_count_r[seed_ind_r] >= 10 or repeat_count_l[seed_ind_l] >= 10:
            #     print("bumping seed!")
            #     seed_ind_r = min(seed_ind_r + 2, aligned_right.shape[0] - 2)
            #     seed_ind_l = min(seed_ind_l + 2, aligned_left.shape[0] - 2)

            # print("seed_ind_r: " + str(seed_ind_r))
            # print("seed_ind_l: " + str(seed_ind_l))

            seed = {}
            seed['right'] = aligned_right[:, :][seed_ind_r, :]
            seed['left'] = aligned_left[:, :][seed_ind_l, :]

            # if self.replan and subplan_number == 1:
            if primitive_name == 'grasp':
                valid = subplan_number == 1
            else:
                valid = True

            both_contact = self.robot.is_in_contact(self.object_id)['right'] and \
                self.robot.is_in_contact(self.object_id)['left']
            if both_contact and not set_time and primitive_name == 'pivot':# and self.replan:
                self.robot.set_sim_sleep(0.35)
                set_time = True

            if set_time and not both_contact:
                self.robot.set_sim_sleep(0.01)

            perturb_valid = both_contact and \
                not perturbed and not perturbing and \
                np.random.random() > 0.9 and \
                seed_ind_r > aligned_right.shape[0]/2.5 and self.perturb #\
                # primitive_name == 'grasp' and subplan_number == 1 and self.perturb
            # perturb_valid = False
            if perturb_valid:
                perturbing = True
                f1 = np.random.random()*3 + 3
                f2 = np.random.random()*3 + 3
                f1 = -f1 if np.random.random() > 0.5 else f1
                f2 = -f2 if np.random.random() > 0.5 else f2
                tau_1 = np.random.random()*4.0 - 2.0
                tau_2 = np.random.random()*4.0 - 2.0
                # tau_2 = np.random.random()*0.5 - 0.25
                # f = 7.75/1.0
                # tau = -0.25/1.0
                # f_vec = [f1, 0, f2] if np.random.random() > 0.5 else [0, f1, f2]
                f_vec = [0, f1, 0]
                # tau_vec = [0, tau_1, 0] if np.random.random() > 0.5 else [0, 0, tau_1]
                tau_vec = [0, 0, tau_1]
                print('\n\n\nApplying perturbation! Force: %f, Torque: %f\n\n\n' % (f1, tau_1))

            if perturbing:
                obj_pos = p.getBasePositionAndOrientation(self.object_id)[0]
                p.applyExternalForce(
                    self.object_id,
                    -1,
                    f_vec,
                    obj_pos,
                    p.WORLD_FRAME)
                p.applyExternalTorque(
                    self.object_id,
                    -1,
                    tau_vec,
                    p.LINK_FRAME
                )
                perturbed_n += 1
                self.perturbations_list.append((f_vec, len(self.object_poses_list)))
                if perturbed_n > 15:
                    perturbed = True
                    perturbing = False
                    print('\n\n\nPerturbation complete!\n\n\n')

            # if primitive_name == 'grasp' and (prev_seed_ind_l == seed_ind_l or prev_seed_ind_r == seed_ind_r) and not self.replan:
            #     seed_ind_r += 1
            #     seed_ind_l += 1

            if self.replan and both_contact and valid:
                self._tactile_in_contact = True
                ik_iter = 0
                ik_sol_found = False
                while not ik_sol_found:
                    joints_execute = self.greedy_replan(
                        primitive_name=primitive_name,
                        object_id=self.object_id,
                        seed=seed,
                        plan_number=subplan_number,
                        frac_done=pos_err/pos_err_total)[0]
                    if joints_execute['right'] is not None and \
                            joints_execute['left'] is not None:
                        ik_sol_found = True
                    ik_iter += 1
                    if ik_iter > self.max_ik_iter:
                        raise ValueError('IK solution not found!')
                both_joints = joints_execute['right'] + joints_execute['left']
                # print('yes')
            else:
                # move to the next point w.r.t. closest point
                r_pos = aligned_right[:, :][max(seed_ind_r, seed_ind_l)+1, :].tolist()
                l_pos = aligned_left[:, :][max(seed_ind_r, seed_ind_l)+1, :].tolist()
                both_joints = r_pos + l_pos

            # r_pos, l_pos = both_joints[:7], both_joints[7:]
            # print(both_joints)
            self.robot.update_joints(both_joints, prev=prev_joints)

            # see how far we are from the goal
            reached_goal, pos_err, ori_err = self.reach_pose_goal(
                subplan_goal[:3],
                subplan_goal[3:],
                self.object_id,
                pos_tol=self.goal_pos_tol, ori_tol=self.goal_ori_tol
            )
            errs.append([pos_err, ori_err])

            timed_out = time.time() - start_time > self.subgoal_timeout
            if timed_out and primitive_name == 'grasp':
                print("Timed out!")
                break
            
            # print('Seeds: ' + str(seed_ind_l) + ', ' + str(seed_ind_r))
            # print('Lengths: ' + str(aligned_left.shape[0]) + ', ' + str(aligned_right.shape[0]))

            if (seed_ind_l == aligned_left.shape[0]-2 or \
                    seed_ind_r == aligned_right.shape[0]-2):
                # print("finished full execution, even if not at goal")
                reached_goal = True
            both_contact = self.robot.is_in_contact(self.object_id)['right'] and \
                self.robot.is_in_contact(self.object_id)['left']
            self._tactile_in_contact = both_contact

            prev_joints = both_joints
            prev_seed_ind_l = seed_ind_l
            prev_seed_ind_r = seed_ind_r
            time.sleep(0.1)

        # if subplan_number < 2:
        # move to the end of the subplan before moving on
        # print("moving to end of subplan")
        reached_goal, pos_err, ori_err = self.reach_pose_goal(
            subplan_goal[:3],
            subplan_goal[3:],
            self.object_id,
            pos_tol=0.1, ori_tol=0.05
        )
        for i in range(max(seed_ind_r, seed_ind_l), aligned_right.shape[0] - 1):
            r_pos = aligned_right[:, :][i, :].tolist()
            l_pos = aligned_left[:, :][i, :].tolist()
            both_joints = r_pos + l_pos
            self.robot.update_joints(both_joints)
            time.sleep(0.1)
        ind1 = np.argmin(np.asarray(errs)[:, 0], axis=0)
        ind2 = np.argmin(np.asarray(errs)[:, 1], axis=0)
        print('MIN ERRS: ', errs[ind1], errs[ind2])            
        return reached_goal, pos_err, ori_err

    def execute(self, primitive_name, execute_args, contact_face=None):
        """
        High level execution function. Sets up internal variables and
        preps for calling self.execute_single_arm or self.execute_two_arm

        Args:
            primitive_name (str): Which primitive to execute
            execute_args (dict): dict containing the necessary arguments for
                each primitive

        Returns:
            3-element tuple containing:
            - bool: Whether the goal was reached or not. True if goal
              reached within desired tolerance, else False
            - float: position error at the end
            - float: orientation error at the end
        """
        if primitive_name not in self.primitives:
            raise ValueError('Primitive not recognized')

        self.object_pose_init = \
            util.pose_stamped2list(execute_args['object_pose1_world'])
        self.object_pose_final = \
            execute_args['object_pose2_world']

        self.table_face = execute_args['table_face']
        self.nominal_palms = self.get_nominal_palms(execute_args)

        self.active_arm, self.inactive_arm = \
            self.get_active_arm(self.object_pose_init)

        if primitive_name == 'pivot' or primitive_name == 'grasp':
            two_arm = True
        else:
            two_arm = False

        self.contact_face = contact_face
        self.transform_mesh_world()

        self.initial_plan = self.get_primitive_plan(
            primitive_name,
            execute_args,
            self.active_arm
        )

        # can check if whole path is feasible here?
        valid_plan = self.full_mp_check(
            self.initial_plan, primitive_name)

        success, pos_err, ori_err = None, None, None

        self.primitive_name = primitive_name
        if not self.tactile_thread_started:
            if primitive_name == 'pivot':
                self._object, self.contact_dict, self.optimization_equilibrium_parameters, self.optimization_control_parameters = initialize_levering_tactile_setup()
            elif primitive_name == 'grasp':
                self._object, self.contact_dict, self.optimization_equilibrium_parameters, self.optimization_control_parameters = initialize_grasping_tactile_setup()
            elif primitive_name == 'pull':
                self._object, self.contact_dict, self.optimization_equilibrium_parameters, self.optimization_control_parameters = initialize_pulling_tactile_setup()
            if primitive_name in ['pull', 'grasp', 'pivot']:
                self.tactile_control = TactileControl(self._object, self.contact_dict, self.optimization_equilibrium_parameters, self.optimization_control_parameters)

        self.table_object_contact_list = []
        self.right_object_contact_list = []
        self.left_object_contact_list = []
        self.vertex_positions_list = []
        self.object_poses_list = []
        self.tip_poses_list = []
        self.perturbations_list = []

        if valid_plan:
            subplan_number = 0
            done = False
            start = time.time()
            # self.tactile_state_estimator.start()
            # if primitive_name == 'pivot':
            # if primitive_name in ['pivot', 'pull', 'grasp']:
            if not self.tactile_thread_started:
                self.tactile_state_estimator.start()
                self.tactile_thread_started = True
                # for pose in self.initial_plan[0]['object_poses_world']:
                #     pose_list = util.pose_stamped2list(pose)
                #     p.resetBasePositionAndOrientation(self.object_id, pose_list[:3], pose_list[3:])
                #     time.sleep(0.01)
                # done = True
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
        else:
            print("Full motion planning failed, plan not valid")
            return None


def visualize_goal_state(object_id, goal_pose, pb_client):
    """
    Function for visualizing the goal state in the background,
    with transparent version of the object

    Args:
        object_id (int): PyBullet object id of transparent object
        goal_pose (list): [x, y, z, x, y, z, w] goal pose of the
            object
        pb_client (int): PyBullet client id to connect to physics
            simulation.
    """
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

    yumi_ar = Robot('yumi_palms',
                    pb=True,
                    pb_cfg={'gui': True, 'realtime': False},
                    arm_cfg={'self_collision': False})

    # yumi_ar = Robot('yumi_palms',
    #                 pb=True,
    #                 pb_cfg={'gui': True, 'realtime': True},
    #                 arm_cfg={'self_collision': False})    

    yumi_ar.arm.set_jpos(cfg.RIGHT_INIT + cfg.LEFT_INIT, ignore_physics=True)

    r_gel_id = cfg.RIGHT_GEL_ID
    l_gel_id = cfg.LEFT_GEL_ID

    alpha = cfg.ALPHA
    K = cfg.GEL_CONTACT_STIFFNESS
    restitution = cfg.GEL_RESTITUION

    # p.changeDynamics(
    #     yumi_ar.arm.robot_id,
    #     r_gel_id,
    #     restitution=restitution,
    #     contactStiffness=K,
    #     contactDamping=alpha*K,
    #     rollingFriction=1.0
    # )

    # p.changeDynamics(
    #     yumi_ar.arm.robot_id,
    #     l_gel_id,
    #     restitution=restitution,
    #     contactStiffness=K,
    #     contactDamping=alpha*K,
    #     rollingFriction=1.0
    # )
    # lateral = 1.3
    lateral = 0.5
    rolling = args.rolling
    spinning = 1e-3
    p.changeDynamics(
        yumi_ar.arm.robot_id,
        r_gel_id,
        lateralFriction=lateral
    )

    p.changeDynamics(
        yumi_ar.arm.robot_id,
        l_gel_id,
        lateralFriction=lateral
    )
    # p.changeDynamics(
    #     yumi_ar.arm.robot_id,
    #     r_gel_id,
    #     contactStiffness=K,
    #     contactDamping=alpha*K,
    #     rollingFriction=rolling,
    #     lateralFriction=lateral
    # )

    # p.changeDynamics(
    #     yumi_ar.arm.robot_id,
    #     l_gel_id,
    #     restitution=restitution,
    #     contactStiffness=K,
    #     contactDamping=alpha*K,
    #     rollingFriction=rolling,
    #     lateralFriction=lateral,
    #     spinningFriction=spinning,
    # )

    # p.changeDynamics(
    #     yumi_ar.arm.robot_id,
    #     cfg.TABLE_ID,
    #     contactStiffness=K*100,
    #     contactDamping=alpha*K*100,
    #     lateralFriction=0.2,
    #     spinningFriction=0.0,
    # )
    # p.changeDynamics(
    #     yumi_ar.arm.robot_id,
    #     cfg.TABLE_ID,
    #     lateralFriction=0.1,
    #     spinningFriction=1e-8,
    #     rollingFriction=0.0)
    # #     contactStiffness=K*100,
    # #     contactDamping=alpha*K*100
    # # )

    p.changeDynamics(
        yumi_ar.arm.robot_id,
        cfg.TABLE_ID,
        lateralFriction=0.1)
    #     contactStiffness=K*100,
    #     contactDamping=alpha*K*100
    # )


    yumi_gs = YumiGelslimPybulet(yumi_ar, cfg)

    if args.object:
        box_id = yumi_ar.pb_client.load_urdf(
            args.config_package_path +
            'descriptions/urdf/'+args.object_name+'.urdf',
            cfg.OBJECT_INIT[0:3],
            cfg.OBJECT_INIT[3:]
        )
        goal_box_id = yumi_ar.pb_client.load_urdf(
            args.config_package_path +
            'descriptions/urdf/realsense_box_trans.urdf',
            cfg.OBJECT_FINAL[0:3],
            cfg.OBJECT_FINAL[3:]
        )
        for jnt_id in range(27-1):
            p.setCollisionFilterPair(yumi_ar.arm.robot_id, goal_box_id, jnt_id, -1, enableCollision=False)

        # p.setCollisionFilterPair(yumi_ar.robot_id,
        #                          goal_box_id,
        #                          gel_id,
        #                          -1,
        #                          enableCollision=False)
        p.setCollisionFilterPair(box_id,
                                 goal_box_id,
                                 -1,
                                 -1,
                                 enableCollision=False)
    # embed()
    p.resetBasePositionAndOrientation(box_id, cfg.OBJECT_INIT[:3], cfg.OBJECT_INIT[3:])
    # p.changeDynamics(
    #     box_id,
    #     -1,
    #     spinningFriction=1e-8
    # )
    # localInertiaDiagonal=[2e-4, 9e-5, 2e-4]
    print(p.getDynamicsInfo(box_id, -1))

    mesh_file = args.config_package_path + \
        'descriptions/meshes/objects/' + args.object_name + '.stl'

    # setup macro_planner
    action_planner = ClosedLoopMacroActions(
        cfg,
        yumi_gs,
        box_id,
        yumi_ar.pb_client.get_client_id(),
        args.config_package_path,
        replan=args.replan,
        contact_face=0,
        object_mesh_file=mesh_file,
        perturb=args.perturb
    )
    # embed()

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
    # example_args['palm_pose_r_object'].pose.position.z -= 0.003
    example_args['object'] = manipulated_object
    # example_args['N'] = 20  # 60
    example_args['N'] = 100
    # example_args['N'] = 60
    example_args['init'] = True
    example_args['table_face'] = 0

    primitive_name = args.primitive



    # embed()
    result = action_planner.execute(primitive_name, example_args)
    print(result)

    data = {}
    data['right_obj'] = action_planner.right_object_contact_list
    data['left_obj'] = action_planner.left_object_contact_list
    data['table_obj'] = action_planner.table_object_contact_list
    data['vertices'] = action_planner.vertex_positions_list
    data['tip_poses'] = action_planner.tip_poses_list
    data['object_poses'] = action_planner.object_poses_list
    data['perturbations'] = action_planner.perturbations_list

    goal_pose_list = util.pose_stamped2list(action_planner.object_pose_final)
    action_planner.transform_mesh_world(pose=goal_pose_list)
    vertices = {}
    for i, vertex in enumerate(action_planner.mesh_world.vertices):
        vertices[i] = vertex

    data['object_goal_pose'] = goal_pose_list
    data['object_goal_vertices'] = vertices

    import pickle
    if args.save:
        fname = args.primitive + '_' + args.exp + '.pkl'
        with open(os.path.join(os.getcwd(), 'data/tro', fname), 'wb') as f:
            pickle.dump(data, f)
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

    parser.add_argument(
        '--exp',
        type=str,
        default='debug')

    parser.add_argument(
        '--save',
        action='store_true'
    )

    parser.add_argument(
        '--perturb',
        action='store_true'
    )

    args = parser.parse_args()
    main(args)
