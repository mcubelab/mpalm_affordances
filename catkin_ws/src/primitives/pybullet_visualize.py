from planning import pushing_planning, grasp_planning, levering_planning, pulling_planning
from helper import util, planning_helper, collisions

import os
from example_config import get_cfg_defaults
import copy

from airobot import Robot
from airobot.utils import pb_util
import time
import argparse
import numpy as np

import pickle

import rospy
from trac_ik_python import trac_ik

data = {}
data['r_palm_pose'] = []
data['r_joint_angles'] = []
data['object_pose'] = []


class IKHelper():
    """
    Class for getting IK solutions for Yumi using
    TRAC-IK ROS package
    """
    def __init__(self):
        """
        Constructor, sets up the internal robot description from
        the ROS parameter server and the numerical IK solver for
        each arm, given a particular base frame and EE frame
        """
        robot_description = '/robot_description'
        urdf_string = rospy.get_param(robot_description)
        self.num_ik_solver_r = trac_ik.IK('yumi_body', 'yumi_tip_r',
                                          urdf_string=urdf_string)

        self.num_ik_solver_l = trac_ik.IK('yumi_body', 'yumi_tip_l',
                                          urdf_string=urdf_string)

    def compute_ik(self, robot, pos, ori, seed, arm='right'):
        """
        Get IK solution given some EE position and orientation,
        for one of Yumi's arms
        
        Args:
            robot (Robot): Instance of PyBullet robot from airobot library
            pos (list or np.ndarray): Desired EE position, [x, y, z]
            ori (list or np.ndarray): Desired EE orientation, in
                quaternion, [x, y, z, w]
            seed (list or np.ndarray): Seed for IK solver. Set of joint
                angle values. Allows user to pass in a known nearby 
                "nice" IK solution.
            arm (str, optional): Which arm to get solution for. 
                Defaults to 'right'.
        
        Returns:
            np.ndarray: Joint angles for the robot to move to for achieving
                desired EE pose. Shape: (1, DOF)
        """
        if arm != 'right' and arm != 'left':
            arm = 'right'
        if arm == 'right':
            sol = self.num_ik_solver_r.get_ik(
                seed,
                pos[0],
                pos[1],
                pos[2],
                ori[0],
                ori[1],
                ori[2],
                ori[3],
                0.01, 0.01, 0.01,
                0.1, 0.1, 0.1
            )
        elif arm == 'left':
            sol = self.num_ik_solver_l.get_ik(
                seed,
                pos[0],
                pos[1],
                pos[2],
                ori[0],
                ori[1],
                ori[2],
                ori[3],
                0.01, 0.01, 0.01,
                0.1, 0.1, 0.1
            )
        return sol


def get_joint_poses(tip_poses, robot, cfg, nullspace=True):
    tip_to_wrist = util.list2pose_stamped(cfg.TIP_TO_WRIST_TF, '')
    world_to_world = util.unit_pose()

    r_joints, l_joints = None, None

    wrist_left = util.convert_reference_frame(
        tip_to_wrist,
        world_to_world,
        tip_poses[0],
        "yumi_body")
    wrist_right = util.convert_reference_frame(
        tip_to_wrist,
        world_to_world,
        tip_poses[1],
        "yumi_body")

    wrist_left = util.pose_stamped2list(wrist_left)
    wrist_right = util.pose_stamped2list(wrist_right)

    r_joints = robot.arm.compute_ik(
        wrist_right[0:3],
        wrist_right[3:],
        arm='right',
        ns=nullspace)

    l_joints = robot.arm.compute_ik(
        wrist_left[0:3],
        wrist_left[3:],
        arm='left',
        ns=nullspace)
    return r_joints, l_joints, wrist_right, wrist_left


def main(args):
    print(args)

    yumi = Robot('yumi',
                 pb=True,
                 arm_cfg={'render': True, 'self_collision': False})
    yumi.arm.go_home()

    cfg_file = os.path.join(args.example_config_path, args.primitive) + ".yaml"
    cfg = get_cfg_defaults()
    cfg.merge_from_file(cfg_file)
    cfg.freeze()
    print(cfg)

    yumi.arm.set_jpos(cfg.RIGHT_INIT+cfg.LEFT_INIT)
    time.sleep(1.0)

    manipulated_object = None
    object_pose1_world = util.list2pose_stamped(cfg.OBJECT_INIT)
    object_pose2_world = util.list2pose_stamped(cfg.OBJECT_FINAL)
    palm_pose_l_object = util.list2pose_stamped(cfg.PALM_LEFT)
    palm_pose_r_object = util.list2pose_stamped(cfg.PALM_RIGHT)

    if args.primitive == 'push':
        plan = pushing_planning(
            object=manipulated_object,
            object_pose1_world=object_pose1_world,
            object_pose2_world=object_pose2_world,
            palm_pose_l_object=palm_pose_l_object,
            palm_pose_r_object=palm_pose_r_object)

    elif args.primitive == 'grasp':
        plan = grasp_planning(
            object=manipulated_object,
            object_pose1_world=object_pose1_world,
            object_pose2_world=object_pose2_world,
            palm_pose_l_object=palm_pose_l_object,
            palm_pose_r_object=palm_pose_r_object)

    elif args.primitive == 'pivot':
        gripper_name = args.config_package_path + \
            'descriptions/meshes/mpalm/mpalms_all_coarse.stl'
        table_name = args.config_package_path + \
            'descriptions/meshes/table/table_top.stl'

        manipulated_object = collisions.CollisionBody(
            args.config_package_path +
            'descriptions/meshes/objects/realsense_box_experiments.stl')

        plan = levering_planning(
            object=manipulated_object,
            object_pose1_world=object_pose1_world,
            object_pose2_world=object_pose2_world,
            palm_pose_l_object=palm_pose_l_object,
            palm_pose_r_object=palm_pose_r_object,
            gripper_name=gripper_name,
            table_name=table_name)

    elif args.primitive == 'pull':
        plan = pulling_planning(
            object=manipulated_object,
            object_pose1_world=object_pose1_world,
            object_pose2_world=object_pose2_world,
            palm_pose_l_object=palm_pose_l_object,
            palm_pose_r_object=palm_pose_r_object,
            arm='r')

    else:
        raise NotImplementedError

    object_loaded = False
    for plan_dict in plan:
        for i, t in enumerate(plan_dict['t']):
            if i == 0 and not object_loaded and args.object:
                time.sleep(2.0)
                pb_util.load_urdf(
                    args.config_package_path +
                    'descriptions/urdf/realsense_box.urdf',
                    cfg.OBJECT_INIT[0:3],
                    cfg.OBJECT_INIT[3:]
                )

                time.sleep(2.0)
                object_loaded = True

            tip_poses = plan_dict['palm_poses_world'][i]

            r_joints, l_joints, wrist_right, wrist_left = get_joint_poses(
                tip_poses,
                yumi,
                cfg,
                nullspace=False)

            loop_time = 0.125
            sleep_time = 0.005
            start = time.time()
            if args.primitive != 'push':
                while (time.time() - start < loop_time):

                    yumi.arm.set_jpos(
                        list(r_joints)+list(l_joints), wait=False)
                    time.sleep(sleep_time)

            else:
                while (time.time() - start < loop_time):
                    yumi.arm.set_jpos(
                        r_joints,
                        arm='right',
                        wait=False)
                    time.sleep(sleep_time)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_package_path',
                        type=str,
                        default='/root/catkin_ws/src/config/')
    parser.add_argument('--example_config_path', type=str, default='config')
    parser.add_argument('--primitive', type=str, default='push', help='which primitive to plan')
    parser.add_argument('--object', action='store_true')
    args = parser.parse_args()
    main(args)
