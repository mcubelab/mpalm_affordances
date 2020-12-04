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

from example_config_cfg import get_cfg_defaults
from yumi_pybullet_ros import YumiGelslimPybullet

class OpenLoopMacroActions(object):
    def __init__(self, cfg, robot, pb=False, pb_info=None):
        self.cfg = cfg
        self.robot = robot
        self.pb = pb
        # we expect to have at object id, mesh file, and pb client info if we're in simulation
        self.pb_info = pb_info
        self._pb_info_keys = ['object_mesh_file', 'object_id', 'pb_client']
        if self.pb:
            assert isinstance(self.pb_info, dict)
            for key in self.pb_info.keys():
                assert key in self._pb_info_keys
            self.object_id = self.pb_info['object_id']
            self.pb_client = self.pb_info['pb_client']
            self.object_mesh_file = self.pb_info['object_mesh_file']

        self.active_arm = 'right'

        self.clear_planning_scene()

    def clear_planning_scene(self):
        # remove all moveit scene objects before we begin
        for name in self.robot.moveit_scene.get_known_object_names():
            self.robot.moveit_scene.remove_world_object(name)

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

    def add_remove_scene_object(self, pose=None, mesh_file=None, object_name='object', action='add'):
        """
        Helper function to add or remove an object from the MoveIt
        planning scene

        Args:
            pose (PoseStamped): Pose in the world frame to put the object
            mesh_file (str): Absolute path to the .stl file to load into the scene
            action (str, optional): Whether to 'add' or 'remove'
                the object. Defaults to 'add'.
        """
        if action != 'add' and action != 'remove':
            raise ValueError('Action not recognied, must be either'
                             'add or remove')

        if (pose is None or mesh_file is None) and self.pb and action=='add':
            object_pos_world = list(p.getBasePositionAndOrientation(
                self.object_id,
                self.pb_client)[0])
            object_ori_world = list(p.getBasePositionAndOrientation(
                self.object_id,
                self.pb_client)[1])
            pose = util.list2pose_stamped(object_pos_world + object_ori_world, 'yumi_body')
            mesh_file = self.object_mesh_file

        if action == 'add':
            pose_stamped = PoseStamped()

            pose_stamped.header.frame_id = pose.header.frame_id
            pose_stamped.pose.position.x = pose.pose.position.x
            pose_stamped.pose.position.y = pose.pose.position.y
            pose_stamped.pose.position.z = pose.pose.position.z
            pose_stamped.pose.orientation.x = pose.pose.orientation.x
            pose_stamped.pose.orientation.y = pose.pose.orientation.y
            pose_stamped.pose.orientation.z = pose.pose.orientation.z
            pose_stamped.pose.orientation.w = pose.pose.orientation.w

            self.robot.moveit_scene.add_mesh(
                name=object_name,
                pose=pose_stamped,
                filename=mesh_file,
                size=(1.01, 1.01, 1.01)
            )
        elif action == 'remove':
            self.robot.moveit_scene.remove_world_object(
                name=object_name
            )

    def single_arm_setup(self, subplan_dict, pre=True):
        """Prepare the system for executing a single arm primitive

        Args:
            subplan_dict (dict): Dictionary containing the primitive plan

        Returns:
            np.ndarray: array of joint configurations corresponding to the path
            np.ndarray: array of palm poses corresponding to the path
        """
        subplan_tip_poses = copy.deepcopy(subplan_dict['palm_poses_world'])

        if pre:
            start_pose_r = util.pose_stamped2list(subplan_tip_poses[0][1])
            start_pose_l = util.pose_stamped2list(subplan_tip_poses[0][0])

            start_poses = {}
            start_poses['right'] = subplan_tip_poses[0][1]
            start_poses['left'] = subplan_tip_poses[0][0]

            # get palm_y_normal
            palm_y_normals = self.robot.get_palm_y_normals(start_poses)

            normal_dir_r = np.asarray(util.pose_stamped2list(palm_y_normals['right']))[:3] - \
                            np.asarray(start_pose_r)[:3]
            normal_dir_l = np.asarray(util.pose_stamped2list(palm_y_normals['left']))[:3] - \
                            np.asarray(start_pose_l)[:3]

            r_pos, r_ori = np.asarray(start_pose_r[:3]), np.asarray(start_pose_r[3:])
            l_pos, l_ori = np.asarray(start_pose_l[:3]), np.asarray(start_pose_l[3:])

            r_pos += normal_dir_r*0.025
            l_pos += normal_dir_l*0.025

            arm = self.active_arm
            r_jnts = self.robot.compute_ik(
                r_pos,
                r_ori,
                self.robot.get_jpos(arm='right'), arm='right')
            l_jnts = self.robot.compute_ik(
                l_pos,
                l_ori,
                self.robot.get_jpos(arm='left'), arm='left')

            if self.pb:
                self.add_remove_scene_object(action='add')
            time.sleep(0.5)
            if arm == 'right':
                l_jnts = self.robot.get_jpos(arm='left')
                if r_jnts is not None:
                    try:
                        joints_r, joints_l = self.robot.move_to_joint_target_mp(list(r_jnts), list(l_jnts))
                    except IndexError:
                        raise ValueError('Hack')
                else:
                    raise ValueError('could not approch')
            else:
                r_jnts = self.robot.get_jpos(arm='right')
                if l_jnts is not None:
                    try:
                        joints_r, joints_l = self.robot.move_to_joint_target_mp(list(r_jnts), list(l_jnts))
                    except IndexError:
                        raise ValueError('Hack')
                else:
                    raise ValueError('could not approch')
            if self.pb:
                self.add_remove_scene_object(action='remove')
            time.sleep(0.5)

            for k in range(joints_r.shape[0]):
                jnts_r = joints_r[k, :]
                jnts_l = joints_l[k, :]
                self.robot.update_joints(jnts_r.tolist() + jnts_l.tolist())
                time.sleep(0.075)

#####################################################################################3

            # # # setup motion planning request with cartesian waypoints
            # tip_right, tip_left = [], []

            # # create an approach waypoint near the object
            # pre_pose_right_init = util.unit_pose()
            # pre_pose_left_init = util.unit_pose()

            # pre_pose_right_init.pose.position.y += 0.05
            # pre_pose_left_init.pose.position.y += 0.05

            # pre_pose_right = util.transform_pose(pre_pose_right_init,
            #                                     subplan_tip_poses[0][1])
            # pre_pose_left = util.transform_pose(pre_pose_left_init,
            #                                     subplan_tip_poses[0][0])
            # tip_right.append(pre_pose_right.pose)
            # tip_left.append(pre_pose_left.pose)

            # tip_right.append(subplan_tip_poses[0][1].pose)
            # tip_left.append(subplan_tip_poses[0][0].pose)

            # # MOVE TO THIS POSITION
            # l_current = self.robot.get_jpos(arm='left')
            # r_current = self.robot.get_jpos(arm='right')
            # # plan cartesian path
            # if self.active_arm == 'right':
            #     joint_traj = self.robot.mp_right.plan_waypoints(
            #         tip_right,
            #         force_start=l_current+r_current,
            #         avoid_collisions=True
            #     )
            # else:
            #     joint_traj = self.robot.mp_left.plan_waypoints(
            #         tip_left,
            #         force_start=l_current+r_current,
            #         avoid_collisions=True
            #     )

            # # make numpy arrays for joints and cartesian points
            # joints_np = np.zeros((len(joint_traj.points), 7))
            # fk_np = np.zeros((len(joint_traj.points), 7))

            # for i, point in enumerate(joint_traj.points):
            #     joints_np[i, :] = point.positions
            #     pose = self.robot.compute_fk(
            #         point.positions,
            #         arm=self.active_arm
            #     )
            #     fk_np[i, :] = util.pose_stamped2list(pose)

            # # follow waypoints in open loop
            # for i in range(joints_np.shape[0]):
            #     joints = joints_np[i, :].tolist()

            #     self.robot.update_joints(joints, arm=self.active_arm)
            #     time.sleep(0.075)

        # setup motion planning request with cartesian waypoints
        tip_right, tip_left = [], []

        # create all other cartesian waypoints
        for i in range(len(subplan_tip_poses)):
            tip_right.append(subplan_tip_poses[i][1].pose)
            tip_left.append(subplan_tip_poses[i][0].pose)

        # robot must be in good initial joint configuration
        l_current = self.robot.get_jpos(arm='left')
        r_current = self.robot.get_jpos(arm='right')
        # plan cartesian path
        if self.active_arm == 'right':
            joint_traj = self.robot.mp_right.plan_waypoints(
                tip_right,
                force_start=l_current+r_current,
                avoid_collisions=True
            )
        else:
            joint_traj = self.robot.mp_left.plan_waypoints(
                tip_left,
                force_start=l_current+r_current,
                avoid_collisions=True
            )

        # make numpy arrays for joints and cartesian points
        joints_np = np.zeros((len(joint_traj.points), 7))
        fk_np = np.zeros((len(joint_traj.points), 7))

        for i, point in enumerate(joint_traj.points):
            joints_np[i, :] = point.positions
            pose = self.robot.compute_fk(
                point.positions,
                arm=self.active_arm
            )
            fk_np[i, :] = util.pose_stamped2list(pose)

        return joints_np, fk_np

    def single_arm_retract(self, arm='right'):
        current_pose_r = util.pose_stamped2list(
            self.robot.compute_fk(self.robot.get_jpos(arm='right'), arm='right'))
        current_pose_l = util.pose_stamped2list(
            self.robot.compute_fk(self.robot.get_jpos(arm='left'), arm='left'))

        # get palm_y_normal
        palm_y_normals = self.robot.get_palm_y_normals()

        normal_dir_r = np.asarray(util.pose_stamped2list(palm_y_normals['right']))[:3] - \
                       np.asarray(current_pose_r)[:3]
        normal_dir_l = np.asarray(util.pose_stamped2list(palm_y_normals['left']))[:3] - \
                       np.asarray(current_pose_l)[:3]

        r_pos, r_ori = np.asarray(current_pose_r[:3]), np.asarray(current_pose_r[3:])
        l_pos, l_ori = np.asarray(current_pose_l[:3]), np.asarray(current_pose_l[3:])
        for _ in range(10):
            r_pos += normal_dir_r*0.001
            l_pos += normal_dir_l*0.001

            r_jnts = self.robot.compute_ik(
                r_pos,
                r_ori,
                self.robot.get_jpos(arm='right'), arm='right')
            l_jnts = self.robot.compute_ik(
                l_pos,
                l_ori,
                self.robot.get_jpos(arm='left'), arm='left')
            if arm == 'right':
                l_jnts = self.robot.get_jpos(arm='left')
                if r_jnts is not None:
                    self.robot.update_joints(list(r_jnts) + list(l_jnts))
            else:
                r_jnts = self.robot.get_jpos(arm='right')
                if l_jnts is not None:
                    self.robot.update_joints(list(r_jnts) + list(l_jnts))
            time.sleep(0.1)

    def single_arm_approach(self, arm='right'):
        current_pose_r = util.pose_stamped2list(
            self.robot.compute_fk(self.robot.get_jpos(arm='right'), arm='right'))
        current_pose_l = util.pose_stamped2list(
            self.robot.compute_fk(self.robot.get_jpos(arm='left'), arm='left'))

        # get palm_y_normal
        palm_y_normals = self.robot.get_palm_y_normals()

        normal_dir_r = np.asarray(util.pose_stamped2list(palm_y_normals['right']))[:3] - \
                       np.asarray(current_pose_r)[:3]
        normal_dir_l = np.asarray(util.pose_stamped2list(palm_y_normals['left']))[:3] - \
                       np.asarray(current_pose_l)[:3]

        r_pos, r_ori = np.asarray(current_pose_r[:3]), np.asarray(current_pose_r[3:])
        l_pos, l_ori = np.asarray(current_pose_l[:3]), np.asarray(current_pose_l[3:])

        r_pos -= normal_dir_r*0.002
        l_pos -= normal_dir_l*0.002

        r_jnts = self.robot.compute_ik(
            r_pos,
            r_ori,
            self.robot.get_jpos(arm='right'), arm='right')
        l_jnts = self.robot.compute_ik(
            l_pos,
            l_ori,
            self.robot.get_jpos(arm='left'), arm='left')
        if arm == 'right':
            l_jnts = self.robot.get_jpos(arm='left')
            if r_jnts is not None:
                self.robot.update_joints(list(r_jnts) + list(l_jnts))
        else:
            r_jnts = self.robot.get_jpos(arm='right')
            if l_jnts is not None:
                self.robot.update_joints(list(r_jnts) + list(l_jnts))

    def dual_arm_setup(self, subplan_dict, subplan_number, pre=True):
        """Prepare the system for executing a dual arm primitive

        Args:
            subplan_dict (dict): Dictionary containing the primitive plan

        Returns:
            np.ndarray: array of joint configurations corresponding to the path
            np.ndarray: array of palm poses corresponding to the path
        """
        subplan_tip_poses = copy.deepcopy(subplan_dict['palm_poses_world'])

        # setup motion planning request with cartesian waypoints
        tip_right, tip_left = [], []

        # create an approach waypoint near the object, if start of motion
        if subplan_number == 0 and pre:
            initial_pose = {}
            initial_pose['right'] = subplan_tip_poses[0][1]
            initial_pose['left'] = subplan_tip_poses[0][0]
            palm_y_normals = self.robot.get_palm_y_normals(palm_poses=initial_pose)
            normal_dir_r = (util.pose_stamped2np(palm_y_normals['right'])[:3] - util.pose_stamped2np(initial_pose['right'])[:3]) * 0.05
            normal_dir_l = (util.pose_stamped2np(palm_y_normals['left'])[:3] - util.pose_stamped2np(initial_pose['left'])[:3]) * 0.05

            pre_pose_right_pos = util.pose_stamped2np(initial_pose['right'])[:3] + normal_dir_r
            pre_pose_left_pos = util.pose_stamped2np(initial_pose['left'])[:3] + normal_dir_l

            pre_pose_right_np = np.hstack([pre_pose_right_pos, util.pose_stamped2np(initial_pose['right'])[3:]])
            pre_pose_left_np = np.hstack([pre_pose_left_pos, util.pose_stamped2np(initial_pose['left'])[3:]])
            pre_pose_right = util.list2pose_stamped(pre_pose_right_np)
            pre_pose_left = util.list2pose_stamped(pre_pose_left_np)

            r_pos, r_ori = pre_pose_right_np[:3], pre_pose_right_np[3:]
            l_pos, l_ori = pre_pose_left_np[:3], pre_pose_left_np[3:]

            # print('setup!')
            # embed()

            # pre_pose_right_init = util.unit_pose()
            # pre_pose_left_init = util.unit_pose()

            # pre_pose_right_init.pose.position.y += 0.05
            # pre_pose_left_init.pose.position.y += 0.05

            # pre_pose_right = util.transform_pose(pre_pose_right_init,
            #                                      subplan_tip_poses[0][1])
            # pre_pose_left = util.transform_pose(pre_pose_left_init,
            #                                     subplan_tip_poses[0][0])

# this is working, but collides
#######################################################################################

            # tip_right.append(pre_pose_right.pose)
            # tip_left.append(pre_pose_left.pose)

            # tip_right.append(subplan_tip_poses[0][1].pose)
            # tip_left.append(subplan_tip_poses[0][0].pose)

            # # robot must be in good initial joint configuration
            # l_current = self.robot.get_jpos(arm='left')
            # r_current = self.robot.get_jpos(arm='right')

            # # l_start = self.robot.get_jpos(arm='left')
            # # r_start = self.robot.get_jpos(arm='right')

            # # plan cartesian paths
            # try:
            #     joint_traj_right = self.robot.mp_right.plan_waypoints(
            #         tip_right,
            #         force_start=l_current+r_current,
            #         avoid_collisions=True
            #     )
            # except ValueError as e:
            #     raise ValueError(e)
            # try:
            #     joint_traj_left = self.robot.mp_left.plan_waypoints(
            #         tip_left,
            #         force_start=l_current+r_current,
            #         avoid_collisions=True
            #     )
            # except ValueError as e:
            #     raise ValueError(e)

            # # after motion planning, unify the dual arm trajectories
            # unified = self.robot.unify_arm_trajectories(
            #     joint_traj_left,
            #     joint_traj_right,
            #     subplan_tip_poses)

            # aligned_left = unified['left']['aligned_joints']
            # aligned_right = unified['right']['aligned_joints']

            # if aligned_left.shape != aligned_right.shape:
            #     raise ValueError('Could not aligned joint trajectories')

            # for i in range(aligned_right.shape[0]):
            #     joints_right = aligned_right[i, :].tolist()
            #     joints_left = aligned_left[i, :].tolist()
            #     both_joints = joints_right + joints_left

            #     self.robot.update_joints(both_joints)
            #     time.sleep(0.075)

# this is experimental
#################################################################################
            r_jnts = self.robot.compute_ik(
                r_pos,
                r_ori,
                self.robot.get_jpos(arm='right'), arm='right')
            l_jnts = self.robot.compute_ik(
                l_pos,
                l_ori,
                self.robot.get_jpos(arm='left'), arm='left')

            if self.pb:
                self.add_remove_scene_object(action='add')
                time.sleep(0.5)

            if r_jnts is not None and l_jnts is not None:
                try:
                    joints_r, joints_l = self.robot.move_to_joint_target_mp(list(r_jnts), list(l_jnts))
                except IndexError:
                    raise ValueError('Hack')
            else:
                raise ValueError('Could not setup, IK failed')

            if self.pb:
                self.add_remove_scene_object(action='remove') 
                time.sleep(0.5)

            for k in range(joints_r.shape[0]):
                jnts_r = joints_r[k, :]
                jnts_l = joints_l[k, :]
                self.robot.update_joints(jnts_r.tolist() + jnts_l.tolist())
                time.sleep(0.075)

        # setup motion planning request with cartesian waypoints
        tip_right, tip_left = [], []

        # create all other cartesian waypoints
        for i in range(len(subplan_tip_poses)):
            tip_right.append(subplan_tip_poses[i][1].pose)
            tip_left.append(subplan_tip_poses[i][0].pose)

        # if at the end of the plan, create a retract waypoint
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

        # robot must be in good initial joint configuration
        l_current = self.robot.get_jpos(arm='left')
        r_current = self.robot.get_jpos(arm='right')

        # plan cartesian paths
        try:
            joint_traj_right = self.robot.mp_right.plan_waypoints(
                tip_right,
                force_start=l_current+r_current,
                avoid_collisions=True
            )
        except ValueError as e:
            raise ValueError(e)
        try:
            joint_traj_left = self.robot.mp_left.plan_waypoints(
                tip_left,
                force_start=l_current+r_current,
                avoid_collisions=True
            )
        except ValueError as e:
            raise ValueError(e)

        # after motion planning, unify the dual arm trajectories
        unified = self.robot.unify_arm_trajectories(
            joint_traj_left,
            joint_traj_right,
            subplan_tip_poses)

        aligned_left = unified['left']['aligned_joints']
        aligned_right = unified['right']['aligned_joints']

        if aligned_left.shape != aligned_right.shape:
            raise ValueError('Could not aligned joint trajectories')

        return unified

    def dual_arm_approach(self, *args, **kwargs):
        current_pose_r = util.pose_stamped2list(
            self.robot.compute_fk(self.robot.get_jpos(arm='right'), arm='right'))
        current_pose_l = util.pose_stamped2list(
            self.robot.compute_fk(self.robot.get_jpos(arm='left'), arm='left'))

        # get palm_y_normal
        palm_y_normals = self.robot.get_palm_y_normals()

        normal_dir_r = np.asarray(util.pose_stamped2list(palm_y_normals['right']))[:3] - \
                       np.asarray(current_pose_r)[:3]
        normal_dir_l = np.asarray(util.pose_stamped2list(palm_y_normals['left']))[:3] - \
                       np.asarray(current_pose_l)[:3]

        r_pos, r_ori = np.asarray(current_pose_r[:3]), np.asarray(current_pose_r[3:])
        l_pos, l_ori = np.asarray(current_pose_l[:3]), np.asarray(current_pose_l[3:])

        r_pos -= normal_dir_r*0.0015
        l_pos -= normal_dir_l*0.0015

        r_jnts = self.robot.compute_ik(
            r_pos,
            r_ori,
            self.robot.get_jpos(arm='right'), arm='right')
        l_jnts = self.robot.compute_ik(
            l_pos,
            l_ori,
            self.robot.get_jpos(arm='left'), arm='left')

        self.robot.update_joints(list(r_jnts) + list(l_jnts))

    def dual_arm_retract(self, *args, **kwargs):
        current_pose_r = util.pose_stamped2list(
            self.robot.compute_fk(self.robot.get_jpos(arm='right'), arm='right'))
        current_pose_l = util.pose_stamped2list(
            self.robot.compute_fk(self.robot.get_jpos(arm='left'), arm='left'))

        # get palm_y_normal
        palm_y_normals = self.robot.get_palm_y_normals()

        normal_dir_r = np.asarray(util.pose_stamped2list(palm_y_normals['right']))[:3] - \
                       np.asarray(current_pose_r)[:3]
        normal_dir_l = np.asarray(util.pose_stamped2list(palm_y_normals['left']))[:3] - \
                       np.asarray(current_pose_l)[:3]

        r_pos, r_ori = np.asarray(current_pose_r[:3]), np.asarray(current_pose_r[3:])
        l_pos, l_ori = np.asarray(current_pose_l[:3]), np.asarray(current_pose_l[3:])
        for _ in range(10):
            r_pos += normal_dir_r*0.0015
            l_pos += normal_dir_l*0.0015

            r_jnts = self.robot.compute_ik(
                r_pos,
                r_ori,
                self.robot.get_jpos(arm='right'), arm='right')
            l_jnts = self.robot.compute_ik(
                l_pos,
                l_ori,
                self.robot.get_jpos(arm='left'), arm='left')

            self.robot.update_joints(list(r_jnts) + list(l_jnts))
            time.sleep(0.1)        

    def playback_single_arm(self, primitive_name, subplan_dict, pre=True):
        """Function to playback an obtained primitive plan purely in open loop,
        with simpler implementation than the full closed loop execution

        Args:
            primitive_name (str): Which primitive to playback
            subplan_dict (dict): The plan to playback
        """
        # get array of waypoints to follow
        # self.add_remove_scene_object('add')
        # time.sleep(0.5)
        joints_np, _ = self.single_arm_setup(subplan_dict, pre=pre)
        # time.sleep(0.5)
        # self.add_remove_scene_object('remove')

        # follow waypoints in open loop
        for i in range(joints_np.shape[0]):
            joints = joints_np[i, :].tolist()

            self.robot.update_joints(joints, arm=self.active_arm)
            time.sleep(0.075)

    def playback_dual_arm(self, primitive_name, subplan_dict,
                          subplan_number, pre=True):
        """Function to playback an obtained primitive plan purely in open loop,
        with simpler implementation than the full closed loop execution

        Args:
            primitive_name (str): Which primitive to playback
            subplan_dict (dict): The plan to playback
            subplan_goal (PoseStamped, optional): Goal state of the object. Defaults to None.
            subplan_number (int, optional): The index of the subplan. Defaults to None.
        """
        # get array of waypoints to follow

        unified = self.dual_arm_setup(subplan_dict, subplan_number, pre=pre)

        aligned_left = unified['left']['aligned_joints']
        aligned_right = unified['right']['aligned_joints']

        for i in range(aligned_right.shape[0]):
            joints_right = aligned_right[i, :].tolist()
            joints_left = aligned_left[i, :].tolist()
            both_joints = joints_right + joints_left

            self.robot.update_joints(both_joints)
            time.sleep(0.075)            


class ClosedLoopMacroActions():
    """
    Class for interfacing with a set of reactive motion primitives
    """
    def __init__(self, cfg, robot, object_id, pb_client,
                 config_pkg_path, object_mesh_file=None, replan=True,
                 contact_face=None):
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

        # self.goal_pos_tol = 0.005  # 0.003
        # self.goal_ori_tol = 0.03  # 0.01
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

        # remove all moveit scene objects before we begin
        for name in self.robot.moveit_scene.get_known_object_names():
            self.robot.moveit_scene.remove_world_object(name)

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

    def transform_mesh_world(self):
        """
        Interal method to transform the object mesh coordinates
        to the world frame, based on where it is in the environment
        """
        self.mesh_world = copy.deepcopy(self.mesh)
        obj_pos_world = list(p.getBasePositionAndOrientation(
            self.object_id, self.pb_client)[0])
        obj_ori_world = list(p.getBasePositionAndOrientation(
            self.object_id, self.pb_client)[1])
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

            pusher_angle = primitive_args['pusher_angle']
            plan = pushing_planning(
                object=manipulated_object,
                object_pose1_world=object_pose1_world,
                object_pose2_world=object_pose2_world,
                palm_pose_l_object=palm_pose_l_object,
                palm_pose_r_object=palm_pose_r_object,
                arm=active_arm[0],
                pusher_angle=pusher_angle)

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
            current_tip_poses['right'].pose.position.z = self.nominal_palms['right']['world'].pose.position.z - 0.002
            current_tip_poses['left'].pose.position.z = self.nominal_palms['left']['world'].pose.position.z - 0.002

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
        primitive_args['table_face'] = self.table_face

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
                        pos_tol=0.01, ori_tol=0.02, z=False):
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
        pos_error = np.linalg.norm(pos_diff)

        quat_diff = common.quat_multiply(common.quat_inverse(goal_ori),
                                         new_ee_quat)
        rot_similarity = np.abs(quat_diff[3])

        if pos_error < pos_tol and \
                rot_similarity > 1 - ori_tol:
            return True, pos_error, 1-rot_similarity
        else:
            return False, pos_error, 1-rot_similarity

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
            self.robot.moveit_scene.add_mesh(
                name='object',
                pose=pose_stamped,
                filename=self.object_mesh_file,
                size=(0.95, 0.95, 0.95)
            )
        elif action == 'remove':
            self.robot.moveit_scene.remove_world_object(
                name='object'
            )

    def full_mp_check(self, initial_plan, primitive_name):
        """
        Run the full plan through motion planning to check feasibility
        right away, don't execute if any of the steps fail motion planning

        Args:
            initial_plan (list): List of plan dictionaries returned
                by primitive planner
            primitive_name (str): Type of primitive being executed,
                determines whether both arms or only single arm should
                be checked

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

            l_start = self.robot.get_jpos(arm='left')
            r_start = self.robot.get_jpos(arm='right')

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
                # print(e)
                # print('Right arm motion planning failed on'
                #       'subplan number %d' % subplan_number)
                break
            try:
                self.robot.mp_left.plan_waypoints(
                    tip_left,
                    force_start=l_start+r_start,
                    avoid_collisions=True
                )
                left_valid.append(1)
            except ValueError as e:
                # print(e)
                # print('Left arm motion planning failed on'
                #       'subplan number %d' % subplan_number)
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

            if primitive_name == 'push':
                # get angle of palm w.r.t object, should be w.r.t normal?
                face_normal = self.get_contact_face_normal()
                if face_normal is not None:
                    palm_y_normal = self.robot.get_palm_y_normals()['right']

                    # compute delta q based on PD control law (want angle to be zero)
                    err = -(1 - np.dot(face_normal, palm_y_normal))
                    dq = self.kp*err + self.kd*(last_err - err)
                    print("dq: " + str(dq))
                    last_err = err

                    # do IK or just rotate the last joint?
                    # joints_execute[-1] += dq
                    joints_update = copy.deepcopy(list(joints_execute))
                    joints_update[-1] += dq
                    joints_execute = tuple(joints_update)

            self.robot.update_joints(joints_execute, arm=self.active_arm)

            reached_goal, pos_err, ori_err = self.reach_pose_goal(
                subplan_goal[:3],
                subplan_goal[3:],
                self.object_id,
                pos_tol=self.goal_pos_tol, ori_tol=self.goal_ori_tol)
            # reached_goal, pos_err, ori_err = self.reach_pose_goal(
            #     subplan_goal[:3],
            #     subplan_goal[3:],
            #     self.object_id,
            #     pos_tol=0.025, ori_tol=0.1)

            timed_out = time.time() - start_time > self.subgoal_timeout
            if timed_out:
                print("TIMED OUT!")
                break
            time.sleep(0.075)
            if not made_contact:
                made_contact = self.robot.is_in_contact(self.object_id)[self.active_arm]
            if made_contact:
                still_in_contact = self.robot.is_in_contact(self.object_id)[self.active_arm]
                slipping = still_in_contact
                if not still_in_contact:
                    slipped += 1
            # if slipped > 15:
            #     print("LOST CONTACT!")
            #     break
        return reached_goal, pos_err, ori_err

    def single_arm_setup(self, subplan_dict, pre=True):
        """Prepare the system for executing a single arm primitive

        Args:
            subplan_dict (dict): Dictionary containing the primitive plan

        Returns:
            np.ndarray: array of joint configurations corresponding to the path
            np.ndarray: array of palm poses corresponding to the path
        """
        subplan_tip_poses = subplan_dict['palm_poses_world']

        if pre:
            # embed()

            # start_pose_r = util.pose_stamped2list(subplan_tip_poses[0][1])
            # start_pose_l = util.pose_stamped2list(subplan_tip_poses[0][0])

            # start_poses = {}
            # start_poses['right'] = subplan_tip_poses[0][1]
            # start_poses['left'] = subplan_tip_poses[0][0]

            # # get palm_y_normal
            # palm_y_normals = self.robot.get_palm_y_normals(start_poses)

            # normal_dir_r = np.asarray(util.pose_stamped2list(palm_y_normals['right']))[:3] - \
            #                 np.asarray(start_pose_r)[:3]
            # normal_dir_l = np.asarray(util.pose_stamped2list(palm_y_normals['left']))[:3] - \
            #                 np.asarray(start_pose_l)[:3]

            # r_pos, r_ori = np.asarray(start_pose_r[:3]), np.asarray(start_pose_r[3:])
            # l_pos, l_ori = np.asarray(start_pose_l[:3]), np.asarray(start_pose_l[3:])

            # r_pos += normal_dir_r*0.05
            # l_pos += normal_dir_l*0.05

            # arm = self.active_arm
            # r_jnts = self.robot.compute_ik(
            #     r_pos,
            #     r_ori,
            #     self.robot.get_jpos(arm='right'), arm='right')
            # l_jnts = self.robot.compute_ik(
            #     l_pos,
            #     l_ori,
            #     self.robot.get_jpos(arm='left'), arm='left')

            # self.add_remove_scene_object('add')
            # time.sleep(0.5)
            # if arm == 'right':
            #     l_jnts = self.robot.get_jpos(arm='left')
            #     if r_jnts is not None:
            #         try:
            #             joints_r, joints_l = self.robot.move_to_joint_target_mp(list(r_jnts), list(l_jnts))
            #         except IndexError:
            #             raise ValueError('Hack')
            #     else:
            #         embed()                    
            #         raise ValueError('could not approch')
            # else:
            #     r_jnts = self.robot.get_jpos(arm='right')
            #     if l_jnts is not None:
            #         try:
            #             joints_r, joints_l = self.robot.move_to_joint_target_mp(list(r_jnts), list(l_jnts))
            #         except IndexError:
            #             raise ValueError('Hack')
            #     else:
            #         embed()
            #         raise ValueError('could not approch')
            # self.add_remove_scene_object('remove')
            # time.sleep(0.5)

            # for k in range(joints_r.shape[0]):
            #     jnts_r = joints_r[k, :]
            #     jnts_l = joints_l[k, :]
            #     self.robot.update_joints(jnts_r.tolist() + jnts_l.tolist())
            #     time.sleep(0.075)


            # # setup motion planning request with cartesian waypoints
            tip_right, tip_left = [], []

            # create an approach waypoint near the object
            pre_pose_right_init = util.unit_pose()
            pre_pose_left_init = util.unit_pose()

            pre_pose_right_init.pose.position.y += 0.05
            pre_pose_left_init.pose.position.y += 0.05

            pre_pose_right = util.transform_pose(pre_pose_right_init,
                                                subplan_tip_poses[0][1])
            pre_pose_left = util.transform_pose(pre_pose_left_init,
                                                subplan_tip_poses[0][0])
            tip_right.append(pre_pose_right.pose)
            tip_left.append(pre_pose_left.pose)

            tip_right.append(subplan_tip_poses[0][1].pose)
            tip_left.append(subplan_tip_poses[0][0].pose)

            # MOVE TO THIS POSITION
            l_current = self.robot.get_jpos(arm='left')
            r_current = self.robot.get_jpos(arm='right')
            # plan cartesian path
            if self.active_arm == 'right':
                joint_traj = self.robot.mp_right.plan_waypoints(
                    tip_right,
                    force_start=l_current+r_current,
                    avoid_collisions=True
                )
            else:
                joint_traj = self.robot.mp_left.plan_waypoints(
                    tip_left,
                    force_start=l_current+r_current,
                    avoid_collisions=True
                )

            # make numpy arrays for joints and cartesian points
            joints_np = np.zeros((len(joint_traj.points), 7))
            fk_np = np.zeros((len(joint_traj.points), 7))

            for i, point in enumerate(joint_traj.points):
                joints_np[i, :] = point.positions
                pose = self.robot.compute_fk(
                    point.positions,
                    arm=self.active_arm
                )
                fk_np[i, :] = util.pose_stamped2list(pose)

            # follow waypoints in open loop
            for i in range(joints_np.shape[0]):
                joints = joints_np[i, :].tolist()

                self.robot.update_joints(joints, arm=self.active_arm)
                time.sleep(0.075)

        # setup motion planning request with cartesian waypoints
        tip_right, tip_left = [], []

        # create all other cartesian waypoints
        for i in range(len(subplan_tip_poses)):
            tip_right.append(subplan_tip_poses[i][1].pose)
            tip_left.append(subplan_tip_poses[i][0].pose)

        # robot must be in good initial joint configuration
        l_current = self.robot.get_jpos(arm='left')
        r_current = self.robot.get_jpos(arm='right')
        # plan cartesian path
        if self.active_arm == 'right':
            joint_traj = self.robot.mp_right.plan_waypoints(
                tip_right,
                force_start=l_current+r_current,
                avoid_collisions=True
            )
        else:
            joint_traj = self.robot.mp_left.plan_waypoints(
                tip_left,
                force_start=l_current+r_current,
                avoid_collisions=True
            )

        # make numpy arrays for joints and cartesian points
        joints_np = np.zeros((len(joint_traj.points), 7))
        fk_np = np.zeros((len(joint_traj.points), 7))

        for i, point in enumerate(joint_traj.points):
            joints_np[i, :] = point.positions
            pose = self.robot.compute_fk(
                point.positions,
                arm=self.active_arm
            )
            fk_np[i, :] = util.pose_stamped2list(pose)

        return joints_np, fk_np

    def single_arm_retract(self, arm='right'):
        current_pose_r = util.pose_stamped2list(
            self.robot.compute_fk(self.robot.get_jpos(arm='right'), arm='right'))
        current_pose_l = util.pose_stamped2list(
            self.robot.compute_fk(self.robot.get_jpos(arm='left'), arm='left'))

        # get palm_y_normal
        palm_y_normals = self.robot.get_palm_y_normals()

        normal_dir_r = np.asarray(util.pose_stamped2list(palm_y_normals['right']))[:3] - \
                       np.asarray(current_pose_r)[:3]
        normal_dir_l = np.asarray(util.pose_stamped2list(palm_y_normals['left']))[:3] - \
                       np.asarray(current_pose_l)[:3]

        r_pos, r_ori = np.asarray(current_pose_r[:3]), np.asarray(current_pose_r[3:])
        l_pos, l_ori = np.asarray(current_pose_l[:3]), np.asarray(current_pose_l[3:])
        for _ in range(10):
            r_pos += normal_dir_r*0.001
            l_pos += normal_dir_l*0.001

            r_jnts = self.robot.compute_ik(
                r_pos,
                r_ori,
                self.robot.get_jpos(arm='right'), arm='right')
            l_jnts = self.robot.compute_ik(
                l_pos,
                l_ori,
                self.robot.get_jpos(arm='left'), arm='left')
            if arm == 'right':
                l_jnts = self.robot.get_jpos(arm='left')
                if r_jnts is not None:
                    self.robot.update_joints(list(r_jnts) + list(l_jnts))
            else:
                r_jnts = self.robot.get_jpos(arm='right')
                if l_jnts is not None:
                    self.robot.update_joints(list(r_jnts) + list(l_jnts))
            time.sleep(0.1)

    def single_arm_approach(self, arm='right'):
        current_pose_r = util.pose_stamped2list(
            self.robot.compute_fk(self.robot.get_jpos(arm='right'), arm='right'))
        current_pose_l = util.pose_stamped2list(
            self.robot.compute_fk(self.robot.get_jpos(arm='left'), arm='left'))

        # get palm_y_normal
        palm_y_normals = self.robot.get_palm_y_normals()

        normal_dir_r = np.asarray(util.pose_stamped2list(palm_y_normals['right']))[:3] - \
                       np.asarray(current_pose_r)[:3]
        normal_dir_l = np.asarray(util.pose_stamped2list(palm_y_normals['left']))[:3] - \
                       np.asarray(current_pose_l)[:3]

        r_pos, r_ori = np.asarray(current_pose_r[:3]), np.asarray(current_pose_r[3:])
        l_pos, l_ori = np.asarray(current_pose_l[:3]), np.asarray(current_pose_l[3:])

        r_pos -= normal_dir_r*0.002
        l_pos -= normal_dir_l*0.002

        r_jnts = self.robot.compute_ik(
            r_pos,
            r_ori,
            self.robot.get_jpos(arm='right'), arm='right')
        l_jnts = self.robot.compute_ik(
            l_pos,
            l_ori,
            self.robot.get_jpos(arm='left'), arm='left')
        if arm == 'right':
            l_jnts = self.robot.get_jpos(arm='left')
            if r_jnts is not None:
                self.robot.update_joints(list(r_jnts) + list(l_jnts))
        else:
            r_jnts = self.robot.get_jpos(arm='right')
            if l_jnts is not None:
                self.robot.update_joints(list(r_jnts) + list(l_jnts))

    def dual_arm_setup(self, subplan_dict, subplan_number, pre=True):
        """Prepare the system for executing a dual arm primitive

        Args:
            subplan_dict (dict): Dictionary containing the primitive plan

        Returns:
            np.ndarray: array of joint configurations corresponding to the path
            np.ndarray: array of palm poses corresponding to the path
        """
        subplan_tip_poses = copy.deepcopy(subplan_dict['palm_poses_world'])

        # setup motion planning request with cartesian waypoints
        tip_right, tip_left = [], []

        # create an approach waypoint near the object, if start of motion
        if subplan_number == 0 and pre:
            initial_pose = {}
            initial_pose['right'] = subplan_tip_poses[0][1]
            initial_pose['left'] = subplan_tip_poses[0][0]
            palm_y_normals = self.robot.get_palm_y_normals(palm_poses=initial_pose)
            normal_dir_r = (util.pose_stamped2np(palm_y_normals['right'])[:3] - util.pose_stamped2np(initial_pose['right'])[:3]) * 0.05
            normal_dir_l = (util.pose_stamped2np(palm_y_normals['left'])[:3] - util.pose_stamped2np(initial_pose['left'])[:3]) * 0.05

            pre_pose_right_pos = util.pose_stamped2np(initial_pose['right'])[:3] + normal_dir_r
            pre_pose_left_pos = util.pose_stamped2np(initial_pose['left'])[:3] + normal_dir_l

            pre_pose_right_np = np.hstack([pre_pose_right_pos, util.pose_stamped2np(initial_pose['right'])[3:]])
            pre_pose_left_np = np.hstack([pre_pose_left_pos, util.pose_stamped2np(initial_pose['left'])[3:]])
            pre_pose_right = util.list2pose_stamped(pre_pose_right_np)
            pre_pose_left = util.list2pose_stamped(pre_pose_left_np)

            # print('setup!')
            # embed()

            # pre_pose_right_init = util.unit_pose()
            # pre_pose_left_init = util.unit_pose()

            # pre_pose_right_init.pose.position.y += 0.05
            # pre_pose_left_init.pose.position.y += 0.05

            # pre_pose_right = util.transform_pose(pre_pose_right_init,
            #                                      subplan_tip_poses[0][1])
            # pre_pose_left = util.transform_pose(pre_pose_left_init,
            #                                     subplan_tip_poses[0][0])

            tip_right.append(pre_pose_right.pose)
            tip_left.append(pre_pose_left.pose)

            tip_right.append(subplan_tip_poses[0][1].pose)
            tip_left.append(subplan_tip_poses[0][0].pose)

            # robot must be in good initial joint configuration
            l_current = self.robot.get_jpos(arm='left')
            r_current = self.robot.get_jpos(arm='right')

            # l_start = self.robot.get_jpos(arm='left')
            # r_start = self.robot.get_jpos(arm='right')

            # plan cartesian paths
            try:
                joint_traj_right = self.robot.mp_right.plan_waypoints(
                    tip_right,
                    force_start=l_current+r_current,
                    avoid_collisions=True
                )
            except ValueError as e:
                raise ValueError(e)
            try:
                joint_traj_left = self.robot.mp_left.plan_waypoints(
                    tip_left,
                    force_start=l_current+r_current,
                    avoid_collisions=True
                )
            except ValueError as e:
                raise ValueError(e)

            # after motion planning, unify the dual arm trajectories
            unified = self.robot.unify_arm_trajectories(
                joint_traj_left,
                joint_traj_right,
                subplan_tip_poses)

            aligned_left = unified['left']['aligned_joints']
            aligned_right = unified['right']['aligned_joints']

            if aligned_left.shape != aligned_right.shape:
                raise ValueError('Could not aligned joint trajectories')

            for i in range(aligned_right.shape[0]):
                joints_right = aligned_right[i, :].tolist()
                joints_left = aligned_left[i, :].tolist()
                both_joints = joints_right + joints_left

                self.robot.update_joints(both_joints)
                time.sleep(0.075)


        # setup motion planning request with cartesian waypoints
        tip_right, tip_left = [], []

        # create all other cartesian waypoints
        for i in range(len(subplan_tip_poses)):
            tip_right.append(subplan_tip_poses[i][1].pose)
            tip_left.append(subplan_tip_poses[i][0].pose)

        # if at the end of the plan, create a retract waypoint
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

        # robot must be in good initial joint configuration
        l_current = self.robot.get_jpos(arm='left')
        r_current = self.robot.get_jpos(arm='right')

        # plan cartesian paths
        try:
            joint_traj_right = self.robot.mp_right.plan_waypoints(
                tip_right,
                force_start=l_current+r_current,
                avoid_collisions=True
            )
        except ValueError as e:
            raise ValueError(e)
        try:
            joint_traj_left = self.robot.mp_left.plan_waypoints(
                tip_left,
                force_start=l_current+r_current,
                avoid_collisions=True
            )
        except ValueError as e:
            raise ValueError(e)

        # after motion planning, unify the dual arm trajectories
        unified = self.robot.unify_arm_trajectories(
            joint_traj_left,
            joint_traj_right,
            subplan_tip_poses)

        aligned_left = unified['left']['aligned_joints']
        aligned_right = unified['right']['aligned_joints']

        if aligned_left.shape != aligned_right.shape:
            raise ValueError('Could not aligned joint trajectories')

        return unified

    def dual_arm_approach(self, arm='right'):
        current_pose_r = util.pose_stamped2list(
            self.robot.compute_fk(self.robot.get_jpos(arm='right'), arm='right'))
        current_pose_l = util.pose_stamped2list(
            self.robot.compute_fk(self.robot.get_jpos(arm='left'), arm='left'))

        # get palm_y_normal
        palm_y_normals = self.robot.get_palm_y_normals()

        normal_dir_r = np.asarray(util.pose_stamped2list(palm_y_normals['right']))[:3] - \
                       np.asarray(current_pose_r)[:3]
        normal_dir_l = np.asarray(util.pose_stamped2list(palm_y_normals['left']))[:3] - \
                       np.asarray(current_pose_l)[:3]

        r_pos, r_ori = np.asarray(current_pose_r[:3]), np.asarray(current_pose_r[3:])
        l_pos, l_ori = np.asarray(current_pose_l[:3]), np.asarray(current_pose_l[3:])

        r_pos -= normal_dir_r*0.0015
        l_pos -= normal_dir_l*0.0015

        r_jnts = self.robot.compute_ik(
            r_pos,
            r_ori,
            self.robot.get_jpos(arm='right'), arm='right')
        l_jnts = self.robot.compute_ik(
            l_pos,
            l_ori,
            self.robot.get_jpos(arm='left'), arm='left')

        self.robot.update_joints(list(r_jnts) + list(l_jnts))

    def playback_single_arm(self, primitive_name, subplan_dict, pre=True):
        """Function to playback an obtained primitive plan purely in open loop,
        with simpler implementation than the full closed loop execution

        Args:
            primitive_name (str): Which primitive to playback
            subplan_dict (dict): The plan to playback
            subplan_goal (PoseStamped, optional): Goal state of the object. Defaults to None.
            subplan_number (int, optional): The index of the subplan. Defaults to None.
        """
        # get array of waypoints to follow
        # self.add_remove_scene_object('add')
        # time.sleep(0.5)
        joints_np, _ = self.single_arm_setup(subplan_dict, pre=pre)
        # time.sleep(0.5)
        # self.add_remove_scene_object('remove')

        # follow waypoints in open loop
        for i in range(joints_np.shape[0]):
            joints = joints_np[i, :].tolist()

            self.robot.update_joints(joints, arm=self.active_arm)
            time.sleep(0.075)

    def playback_dual_arm(self, primitive_name, subplan_dict,
                          subplan_number, pre=True):
        """Function to playback an obtained primitive plan purely in open loop,
        with simpler implementation than the full closed loop execution

        Args:
            primitive_name (str): Which primitive to playback
            subplan_dict (dict): The plan to playback
            subplan_goal (PoseStamped, optional): Goal state of the object. Defaults to None.
            subplan_number (int, optional): The index of the subplan. Defaults to None.
        """
        # get array of waypoints to follow
        unified = self.dual_arm_setup(subplan_dict, subplan_number, pre=pre)
        aligned_left = unified['left']['aligned_joints']
        aligned_right = unified['right']['aligned_joints']

        for i in range(aligned_right.shape[0]):
            joints_right = aligned_right[i, :].tolist()
            joints_left = aligned_left[i, :].tolist()
            both_joints = joints_right + joints_left

            self.robot.update_joints(both_joints)
            time.sleep(0.075)

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
            z=True)

        pos_err = pos_err_total
        ori_err = ori_err_total

        timed_out = False
        start_time = time.time()

        ended = False

        # last_seed_r = -1
        # last_seed_l = -1
        # repeat_count_r = [0] * unified['right']['aligned_fk'].shape[0]
        # repeat_count_l = copy.deepcopy(repeat_count_r)
        while not reached_goal and not ended:
            # check if replanning or not
            # print("starting execution of subplan number: " + str(subplan_number))

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
            both_contact = self.robot.is_in_contact(self.object_id)['right'] and \
                self.robot.is_in_contact(self.object_id)['left']
            if self.replan and both_contact:
                # print("replanning!")
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
            else:
                # move to the next point w.r.t. closest point
                r_pos = aligned_right[:, :][max(seed_ind_r, seed_ind_l)+1, :].tolist()
                l_pos = aligned_left[:, :][max(seed_ind_r, seed_ind_l)+1, :].tolist()
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
            if not self.replan and (seed_ind_l == aligned_left.shape[0]-2 or \
                    seed_ind_r == aligned_right.shape[0]-2):
                # print("finished full execution, even if not at goal")
                reached_goal = True
            both_contact = self.robot.is_in_contact(self.object_id)['right'] and \
                self.robot.is_in_contact(self.object_id)['left']

            time.sleep(0.01)

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

        if valid_plan:
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
        else:
            # print("Full motion planning failed, plan not valid")
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
                    pb_cfg={'gui': True},
                    arm_cfg={'self_collision': False})

    yumi_ar.arm.set_jpos(cfg.RIGHT_INIT + cfg.LEFT_INIT)

    r_gel_id = cfg.RIGHT_GEL_ID
    l_gel_id = cfg.LEFT_GEL_ID

    alpha = cfg.ALPHA
    K = cfg.GEL_CONTACT_STIFFNESS
    restitution = cfg.GEL_RESTITUION

    p.changeDynamics(
        yumi_ar.arm.robot_id,
        r_gel_id,
        restitution=restitution,
        contactStiffness=K,
        contactDamping=alpha*K,
        rollingFriction=args.rolling
    )

    p.changeDynamics(
        yumi_ar.arm.robot_id,
        l_gel_id,
        restitution=restitution,
        contactStiffness=K,
        contactDamping=alpha*K,
        rollingFriction=args.rolling
    )

    yumi_gs = YumiGelslimPybullet(yumi_ar, cfg)

    if args.object:
        box_id = yumi_ar.pb_client.load_urdf(
            args.config_package_path +
            'descriptions/urdf/'+args.object_name+'.urdf',
            cfg.OBJECT_INIT[0:3],
            cfg.OBJECT_INIT[3:]
        )

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
        object_mesh_file=mesh_file
    )
    embed()

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
    example_args['table_face'] = 0

    primitive_name = args.primitive

    result = action_planner.execute(primitive_name, example_args)


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
