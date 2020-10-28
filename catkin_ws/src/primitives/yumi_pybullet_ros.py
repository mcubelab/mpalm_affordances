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
# from airobot.utils import pb_util, common
# from airobot.utils.pb_util import step_simulation

from example_config_cfg import get_cfg_defaults

# from relaxed_ik.msg import EEPoseGoals, JointAngles


class YumiGelslimPybullet(object):
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
        if exec_thread:
            self.execute_thread.start()
            self.step_sim_mode = False
        else:
            self.step_sim_mode = True

        # relaxed IK pub/sub
        # self.ik_lock = threading.RLock()
        # self.relaxed_publisher = rospy.Publisher('/relaxed_ik/ee_pose_goals', EEPoseGoals, queue_size=10)
        # self.relaxed_subscriber = rospy.Subscriber('/relaxed_ik/joint_angle_solutions', JointAngles, self._relaxed_cb)
        # self._relaxed_angles = self.yumi_pb.arm._home_position

    # def _relaxed_cb(self, data):
    #     self.ik_lock.acquire()
    #     self._relaxed_angles = data.angles
    #     self.ik_lock.release()

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

    # def compute_ik(self, pos, ori, seed, arm='right', *args, **kwargs):
    #     if arm == 'right':
    #         r_pos, r_ori = pos, ori
    #         l_pos, l_ori = self.get_ee_pose(arm='left')[0], self.get_ee_pose(arm='left')[1]
    #     else:
    #         l_pos, l_ori = pos, ori
    #         r_pos, r_ori = self.get_ee_pose(arm='right')[0], self.get_ee_pose(arm='right')[1]
    #     goal_msg = EEPoseGoals()
    #     r_pose = Pose()
    #     r_pose.position.x = pos[0]
    #     r_pose.position.y = pos[1]
    #     r_pose.position.z = pos[2]
    #     r_pose.orientation.x = ori[0]
    #     r_pose.orientation.y = ori[1]
    #     r_pose.orientation.z = ori[2]
    #     r_pose.orientation.w = ori[3]

    #     l_pose = Pose()
    #     l_pose.position.x = pos[0]
    #     l_pose.position.y = pos[1]
    #     l_pose.position.z = pos[2]
    #     l_pose.orientation.x = ori[0]
    #     l_pose.orientation.y = ori[1]
    #     l_pose.orientation.z = ori[2]
    #     l_pose.orientation.w = ori[3]
    #     goal_msg.ee_poses.append(r_pose)
    #     goal_msg.ee_poses.append(l_pose)
    #     self.relaxed_publisher.publish(goal_msg)

    #     self.ik_lock.acquire()
    #     sol = self._relaxed_angles
    #     self.ik_lock.release()

    #     if not isinstance(sol, list):
    #         sol = list(sol.data)
    #     sol = sol[:7] if arm == 'right' else sol[7:]

    #     return sol

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
            bodyA=self.yumi_pb.arm.robot_id, bodyB=object_id, linkIndexA=self.cfg.RIGHT_GEL_ID, physicsClientId=self.yumi_pb.pb_client.get_client_id())
        l_pts = p.getContactPoints(
            bodyA=self.yumi_pb.arm.robot_id, bodyB=object_id, linkIndexA=self.cfg.LEFT_GEL_ID, physicsClientId=self.yumi_pb.pb_client.get_client_id())

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

    def get_palm_y_normals(self, palm_poses=None, *args):
        """
        Gets the updated world frame normal direction of the palms
        """
        normal_y = util.list2pose_stamped([0, 1, 0, 0, 0, 0, 1])

        normal_y_poses_world = {}
        wrist_poses = {}
        tip_poses = {}

        for arm in ['right', 'left']:
            if palm_poses is None:
                wrist_pos_world = self.get_ee_pose(arm=arm)[0].tolist()
                wrist_ori_world = self.get_ee_pose(arm=arm)[1].tolist()

                wrist_poses[arm] = util.list2pose_stamped(wrist_pos_world + wrist_ori_world)
            else:
                # wrist_poses[arm] = palm_poses[arm]
                tip_poses[arm] = palm_poses[arm]

        if palm_poses is None:
            tip_poses = self.wrist_to_tip(wrist_poses)

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
        tip_poses = {}

        for arm in ['right', 'left']:
            if palm_poses is None:
                wrist_pos_world = self.get_ee_pose(arm=arm)[0].tolist()
                wrist_ori_world = self.get_ee_pose(arm=arm)[1].tolist()

                wrist_poses[arm] = util.list2pose_stamped(wrist_pos_world + wrist_ori_world)
            else:
                # wrist_poses[arm] = palm_poses[arm]
                tip_poses[arm] = palm_poses[arm]

        if palm_poses is None:
            tip_poses = self.wrist_to_tip(wrist_poses)

        normal_z_poses_world['right'] = util.transform_pose(normal_z, tip_poses['right'])
        normal_z_poses_world['left'] = util.transform_pose(normal_z, tip_poses['left'])

        return normal_z_poses_world

    def get_current_tip_poses(self):

        wrist_poses = {}

        for arm in ['right', 'left']:
            wrist_pos_world = self.get_ee_pose(arm=arm)[0].tolist()
            wrist_ori_world = self.get_ee_pose(arm=arm)[1].tolist()

            wrist_poses[arm] = util.list2pose_stamped(wrist_pos_world + wrist_ori_world)

        tip_poses = self.wrist_to_tip(wrist_poses)

        return tip_poses

    def move_to_joint_target_mp(self, r_jnts, l_jnts, execute=False):
        # set start state to current state
        l_current = self.get_jpos(arm='left')
        r_current = self.get_jpos(arm='right')

        self.mp_right.set_start_state(l_current+r_current)
        self.mp_left.set_start_state(l_current+r_current)

        # set joint value target
        self.mp_right.planning_group.set_joint_value_target(r_jnts)
        self.mp_left.planning_group.set_joint_value_target(l_jnts)

        # get plans
        r_plan = self.mp_right.planning_group.plan()
        l_plan = self.mp_left.planning_group.plan()

        joint_traj_right = r_plan.joint_trajectory
        joint_traj_left = l_plan.joint_trajectory

        left_arm = joint_traj_left
        right_arm = joint_traj_right

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

        if long_traj == 'left':
            new_right = np.zeros((left_arm_joints_np.shape))

            # new_right the same as old right, up to the index that old_right fills
            new_right[:right_arm_joints_np.shape[0], :] = right_arm_joints_np

            # pad the end with the same value as the final joints
            new_right[right_arm_joints_np.shape[0]:, :] = right_arm_joints_np[-1, :]

            aligned_right_joints = new_right
            aligned_left_joints = left_arm_joints_np
        else:
            new_left = np.zeros((right_arm_joints_np.shape))

            # new_right the same as old right, up to the index that old_right fills
            new_left[:left_arm_joints_np.shape[0], :] = left_arm_joints_np

            # pad the end with the same value as the final joints
            new_left[left_arm_joints_np.shape[0]:, :] = left_arm_joints_np[-1, :]

            aligned_right_joints = right_arm_joints_np
            aligned_left_joints = new_left

        if execute:
            for k in range(aligned_right_joints.shape[0]):
                jnts_r = aligned_right_joints[k, :]
                jnts_l = aligned_left_joints[k, :]
                self.yumi_pb.arm.set_jpos(jnts_r.tolist() + jnts_l.tolist(), wait=True)
                time.sleep(0.01)

        return aligned_right_joints, aligned_left_joints