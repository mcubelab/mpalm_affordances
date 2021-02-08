import os
import time
import argparse
import numpy as np
import copy
import threading
import trimesh

from geometry_msgs.msg import PoseStamped

import pybullet as p
from airobot import Robot
from airobot.utils import pb_util, common

from yumi_pybullet_ros import YumiGelslimPybullet
from rpo_planning.utils import common as util


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


class OpenLoopMacroActionsReal(object):
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
        self._loop_t = 0.025
        # self.clear_planning_scene()


    def set_loop_t(self, loop_t):
        self._loop_t = loop_t


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

        if pose is not None and mesh_file is not None:
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
                    size=(0.95, 0.95, 0.95)
                )
            elif action == 'remove':
                self.robot.moveit_scene.remove_world_object(
                    name=object_name
                )
        else:
            raise ValueError('Must provide a mesh file and pose to load into the planning scene')

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
            #################### IK VERSION WORKS BUT IK SOMETIMES DOES HORRIBLE THINGS #########################
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

            # # r_pos += normal_dir_r*0.025
            # # l_pos += normal_dir_l*0.025
            # r_pos += normal_dir_r*0.04
            # l_pos += normal_dir_l*0.04            

            # arm = self.active_arm
            # r_jnts = self.robot.compute_ik(
            #     r_pos,
            #     r_ori,
            #     self.robot.get_jpos(arm='right'), arm='right')
            # l_jnts = self.robot.compute_ik(
            #     l_pos,
            #     l_ori,
            #     self.robot.get_jpos(arm='left'), arm='left')

            # if self.pb:
            #     self.add_remove_scene_object(action='add')
            # time.sleep(0.5)
            # if arm == 'right':
            #     l_jnts = self.robot.get_jpos(arm='left')
            #     if r_jnts is not None:
            #         try:
            #             joints_r, joints_l = self.robot.move_to_joint_target_mp(list(r_jnts), list(l_jnts))
            #         except ValueError as e:
            #             raise ValueError(e)
            #     else:
            #         raise ValueError('could not approch')
            # else:
            #     r_jnts = self.robot.get_jpos(arm='right')
            #     if l_jnts is not None:
            #         try:
            #             joints_r, joints_l = self.robot.move_to_joint_target_mp(list(r_jnts), list(l_jnts))
            #         except ValueError as e:
            #             raise ValueError(e)
            #     else:
            #         raise ValueError('could not approch')
            # if self.pb:
            #     self.add_remove_scene_object(action='remove')

            # print('Got setup motion plan, ready to move joints')

            # # self.robot.yumi_ar.arm.right_arm.set_jpos_buffer(joints_r, sync=True, execute=True, wait=False)
            # # self.robot.yumi_ar.arm.left_arm.set_jpos_buffer(joints_l, sync=True, execute=True, wait=False)

            # joints_r = util.interpolate_joint_trajectory(joints_r, N=200)
            # joints_l = util.interpolate_joint_trajectory(joints_l, N=200)
            # for k in range(joints_r.shape[0]):
            #     jnts_r = joints_r[k, :]
            #     jnts_l = joints_l[k, :]
            #     self.robot.update_joints(jnts_r.tolist() + jnts_l.tolist())
            #     time.sleep(self._loop_t)

            # # make sure robot actually gets to start pose
            # time.sleep(0.5)
            # self.robot.yumi_ar.arm.set_jpos(joints_r[-1, :].tolist() + joints_l[-1, :].tolist(), wait=True)            

            ############################## OLD VERSION THAT USES PLAN_WAYPOINTS INSTEAD OF IK #########################
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

            joints_np = util.interpolate_joint_trajectory(joints_np, N=200)
            # follow waypoints in open loop
            for i in range(joints_np.shape[0]):
                joints = joints_np[i, :].tolist()

                self.robot.update_joints(joints, arm=self.active_arm)
                time.sleep(self._loop_t)

            time.sleep(0.5)
            self.robot.yumi_ar.arm.set_jpos(joints_np[-1, :].tolist(), arm=self.active_arm, wait=True)            

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

    def single_arm_retract(self, arm='right', repeat=1, egm=True):
        for _ in range(repeat):
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
                        self.robot.update_joints(list(r_jnts) + list(l_jnts), egm=egm)
                else:
                    r_jnts = self.robot.get_jpos(arm='right')
                    if l_jnts is not None:
                        self.robot.update_joints(list(r_jnts) + list(l_jnts), egm=egm)
                time.sleep(0.1)

    def single_arm_approach(self, arm='right', repeat=1, egm=True):
        for _ in range(repeat):
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
                    self.robot.update_joints(list(r_jnts) + list(l_jnts), egm=egm)
            else:
                r_jnts = self.robot.get_jpos(arm='right')
                if l_jnts is not None:
                    self.robot.update_joints(list(r_jnts) + list(l_jnts), egm=egm)

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

            print('Ready for setup execution')
            # self.robot.yumi_ar.arm.right_arm.set_jpos_buffer(aligned_right, sync=True, wait=False)
            # self.robot.yumi_ar.arm.left_arm.set_jpos_buffer(aligned_left, sync=True, wait=False)
            aligned_right = util.interpolate_joint_trajectory(aligned_right, N=200)
            aligned_left = util.interpolate_joint_trajectory(aligned_left, N=200)
            for i in range(aligned_right.shape[0]):
                joints_right = aligned_right[i, :].tolist()
                joints_left = aligned_left[i, :].tolist()
                both_joints = joints_right + joints_left

                self.robot.update_joints(both_joints)
                time.sleep(self._loop_t)

            ### maybe not so good on the real robot ###
            # r_jnts = self.robot.compute_ik(
            #     r_pos,
            #     r_ori,
            #     self.robot.get_jpos(arm='right'), arm='right')
            # l_jnts = self.robot.compute_ik(
            #     l_pos,
            #     l_ori,
            #     self.robot.get_jpos(arm='left'), arm='left')

            # if self.pb:
            #     self.add_remove_scene_object(action='add')
            #     time.sleep(0.5)

            # if r_jnts is not None and l_jnts is not None:
            #     try:
            #         joints_r, joints_l = self.robot.move_to_joint_target_mp(list(r_jnts), list(l_jnts))
            #     except IndexError:
            #         raise ValueError('Hack')
            # else:
            #     raise ValueError('Could not setup, IK failed')

            # if self.pb:
            #     self.add_remove_scene_object(action='remove') 
            #     time.sleep(0.5)

            # for k in range(joints_r.shape[0]):
            #     jnts_r = joints_r[k, :]
            #     jnts_l = joints_l[k, :]
            #     self.robot.update_joints(jnts_r.tolist() + jnts_l.tolist())
            #     time.sleep(0.1)
            
            time.sleep(0.5)
            self.robot.yumi_ar.arm.set_jpos(aligned_right[-1, :].tolist() + aligned_left[-1, :].tolist(), wait=True)                

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

    def dual_arm_approach(self, repeat=1, egm=True, *args, **kwargs):
        for _ in range(repeat):
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

            self.robot.update_joints(list(r_jnts) + list(l_jnts), egm=egm)

    def dual_arm_retract(self, repeat=1, egm=True, *args, **kwargs):
        for _ in range(repeat):
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

                self.robot.update_joints(list(r_jnts) + list(l_jnts), egm=egm)
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

        # joints_np = util.interpolate_joint_trajectory(joints_np, N=200)
        # follow waypoints in open loop
        for i in range(joints_np.shape[0]):
            joints = joints_np[i, :].tolist()

            self.robot.update_joints(joints, arm=self.active_arm)
            time.sleep(self._loop_t)
        # if self.active_arm == 'right':
        #     self.robot.yumi_ar.arm.right_arm.set_jpos_buffer(joints_np, sync=False, wait=False)
        # else:
        #     self.robot.yumi_ar.arm.left_arm.set_jpos_buffer(joints_np, sync=False, wait=False)

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

        # aligned_left = util.interpolate_joint_trajectory(aligned_left, N=200)
        # aligned_right = util.interpolate_joint_trajectory(aligned_right, N=200)
        for i in range(aligned_right.shape[0]):
            joints_right = aligned_right[i, :].tolist()
            joints_left = aligned_left[i, :].tolist()
            both_joints = joints_right + joints_left

            self.robot.update_joints(both_joints)
            time.sleep(self._loop_t)    
        # self.robot.yumi_ar.arm.right_arm.set_jpos_buffer(aligned_right, sync=True, wait=False)
        # self.robot.yumi_ar.arm.left_arm.set_jpos_buffer(aligned_left, sync=True, wait=False)

